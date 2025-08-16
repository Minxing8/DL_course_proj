import argparse
import os
import random
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import csv  # Import csv module

from torchvision import transforms

# MiniGPT-4 specific imports
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# Imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Define the same image preprocessing pipeline as in i2t
vis_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str

        image_processed = vis_processor(original_tuple[0])

        return image_processed, path

if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser(description="Adversarial Image Generation for MiniGPT-4 i2t")
    
    # MiniGPT-4 specific arguments
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    
    # Adversarial attack parameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_samples", default=2000, type=int)
    parser.add_argument("--alpha", default=1.0, type=float, help="Step size for perturbation.")
    parser.add_argument("--epsilon", default=16, type=int, help="Maximum perturbation (in pixel values).")
    parser.add_argument("--steps", default=10, type=int, help="Number of attack iterations.")
    parser.add_argument("--output", default="adv_images_minigpt4", type=str, help="Folder to save adversarial images.")
    parser.add_argument("--image_path", required=True, type=str, help="Path to clean images.")
    parser.add_argument("--target_path", required=True, type=str, help="Path to target images.")
    parser.add_argument("--csv_path", required=True, type=str, help="Path to save the mapping CSV file.")  # New argument
    
    # Added `--options` argument to prevent AttributeError
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecated), "
             "use --cfg-options instead.",
        default=None
    )
    
    args = parser.parse_args()
    
    # Adjust alpha and epsilon for normalized images
    alpha = args.alpha / 127.5
    epsilon = args.epsilon / 127.5
    
    # Load MiniGPT-4 model
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model = model.float()  # Convert model parameters to float32
    model.eval()  # Set model to evaluation mode
    
    # Prepare data loaders
    clean_data = ImageFolderWithPaths(args.image_path, transform=None)
    target_data = ImageFolderWithPaths(args.target_path, transform=None)
    data_loader_clean = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    target_loader = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # Define inverse normalization for saving images
    inverse_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )
    
    # Precompute target image features and paths
    print("Precomputing target image features...")
    all_target_images = []
    all_target_paths = []
    
    for image_tgt, paths_tgt in target_loader:
        image_tgt = image_tgt.to(device).to(torch.float32)
        all_target_images.append(image_tgt)
        all_target_paths.extend(paths_tgt)
    
    all_target_images = torch.cat(all_target_images, dim=0)
    num_target_images = all_target_images.size(0)
    print(f"Number of target images: {num_target_images}")
    
    with torch.no_grad():
        tgt_image_embeds = model.ln_vision(model.visual_encoder(all_target_images))
        tgt_image_features = tgt_image_embeds / tgt_image_embeds.norm(dim=-1, keepdim=True)
    
    # Open CSV file and write header
    csv_file = open(args.csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['clean_path', 'target_file'])
    
    # Actual number of samples
    total_clean_images = len(clean_data)
    actual_num_samples = min(args.num_samples, total_clean_images)
    print(f"Actual number of samples to process: {actual_num_samples}")
    
    processed_samples = 0
    os.makedirs(args.output, exist_ok=True)
    
    for i, (image_org, paths_org) in enumerate(tqdm(data_loader_clean, desc="Processing batches")):
        if processed_samples >= actual_num_samples:
            break

        # Calculate current batch size
        current_batch_size = min(args.batch_size, actual_num_samples - processed_samples)

        # Slice current batch and keep as float32
        image_org = image_org[:current_batch_size].to(device).to(torch.float32)
        paths_org = paths_org[:current_batch_size]

        # Compute target indices to cycle through target images
        target_indices = [(processed_samples + idx) % num_target_images for idx in range(current_batch_size)]
        target_indices_tensor = torch.tensor(target_indices, device=device)

        # Get the batch of target images and features
        image_tgt_batch = all_target_images[target_indices_tensor]
        tgt_image_features_batch = tgt_image_features[target_indices_tensor]

        processed_samples += current_batch_size

        # Initialize perturbation, keep as float32 and requires_grad=True
        delta = torch.zeros_like(image_org, requires_grad=True).to(device)

        # Use normal gradient calculation
        for step in range(args.steps):
            # Compute adversarial image
            adv_image = image_org + delta
            adv_image = torch.clamp(adv_image, -1.0, 1.0)  # Keep in [-1, 1] range

            # Forward pass
            adv_image_embeds = model.ln_vision(model.visual_encoder(adv_image))
            adv_image_features = adv_image_embeds / adv_image_embeds.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(adv_image_features, tgt_image_features_batch, dim=-1)
            embedding_sim = cosine_sim.mean()

            # Loss function: maximize similarity (minimize negative similarity)
            loss = embedding_sim
            loss.backward()

            # Check if delta.grad exists
            if delta.grad is None:
                print("Warning: delta.grad is None. Skipping update.")
                break

            # Update perturbation (in-place)
            with torch.no_grad():
                grad_sign = delta.grad.sign()
                delta.add_(alpha * grad_sign)  # In-place update
                delta.clamp_(-epsilon, epsilon)  # In-place clamp
                # Keep adversarial image in [-1, 1] range
                delta.data = (image_org + delta).clamp(-1.0, 1.0) - image_org

            # Zero out gradients
            delta.grad.zero_()

            print(f"Batch {i+1}, Step {step+1}/{args.steps}, Cosine Similarity={embedding_sim.item():.5f}")

        # Generate adversarial image
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image, -1.0, 1.0)  # Keep in normalized range

        # Inverse normalize image for saving
        adv_image_de = inverse_normalize(adv_image)
        adv_image_de = torch.clamp(adv_image_de, 0.0, 1.0)

        # Save adversarial images and record mappings
        for idx in range(len(paths_org)):
            rel_path = os.path.relpath(paths_org[idx], args.image_path)
            rel_dir = os.path.dirname(rel_path)
            folder_to_save = os.path.join(args.output, rel_dir)
            os.makedirs(folder_to_save, exist_ok=True)
            name = os.path.basename(paths_org[idx])
            save_path = os.path.join(folder_to_save, os.path.splitext(name)[0] + '.png')
            torchvision.utils.save_image(adv_image_de[idx].to(torch.float32), save_path)
            print(f"Saved adversarial image to {save_path}")

            # Get the target image file name
            target_index = target_indices[idx]
            target_path = all_target_paths[target_index]
            target_name = os.path.basename(target_path)

            # Write to CSV
            csv_writer.writerow([os.path.abspath(paths_org[idx]), target_name])

    # Close the CSV file
    csv_file.close()

    print("Adversarial attack completed. Adversarial images saved.")
