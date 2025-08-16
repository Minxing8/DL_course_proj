import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import itertools
import shutil

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

def seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # Not needed since each process uses one GPU
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

def partition_indices(total_size, world_size, rank):
    """
    Distribute indices among processes to ensure each process handles a unique subset.
    """
    indices = list(range(total_size))
    per_rank = total_size // world_size
    remainder = total_size % world_size

    if rank < remainder:
        start = rank * (per_rank + 1)
        end = start + (per_rank + 1)
    else:
        start = rank * per_rank + remainder
        end = start + per_rank

    return indices[start:end] if start < end else []

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Adversarial Image Generation for MiniGPT-4 i2t with DDP")

    # MiniGPT-4 specific arguments
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model. (Unused in DDP)")

    # Adversarial attack parameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_samples", default=2000, type=int)  # Adjust as needed
    parser.add_argument("--alpha", default=1.0, type=float, help="Step size for perturbation.")
    parser.add_argument("--epsilon", default=16, type=int, help="Maximum perturbation (in pixel values).")
    parser.add_argument("--steps", default=10, type=int, help="Number of attack iterations.")
    parser.add_argument("--output", default="adv_images_minigpt4", type=str, help="Folder to save adversarial images.")
    parser.add_argument("--image_path", required=True, type=str, help="Path to clean images.")
    parser.add_argument("--target_path", required=True, type=str, help="Path to target images.")

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

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Seed for reproducibility
    seed_everything()

    # Adjust alpha and epsilon for normalized images
    alpha = args.alpha / 127.5
    epsilon = args.epsilon / 127.5

    # Load MiniGPT-4 model
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = local_rank  # Adjust based on your setup
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model = model.float()  # Convert model parameters to float32
    model.eval()  # Set model to evaluation mode

    # Remove the wrapping with DDP
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Debug: Check model parameters' data types (only rank 0)
    if rank == 0:
        for name, param in model.named_parameters():
            print(f"{name}: {param.dtype}")

    # Prepare data loaders by partitioning clean data among processes
    clean_data = ImageFolderWithPaths(args.image_path, transform=None)
    target_data = ImageFolderWithPaths(args.target_path, transform=None)

    # Ensure that num_samples does not exceed the dataset size
    num_clean_samples = min(args.num_samples, len(clean_data))
    if rank == 0:
        print(f"Total clean images to process: {num_clean_samples}")

    # Partition indices uniquely for each process
    assigned_indices = partition_indices(num_clean_samples, world_size, rank)
    clean_subset = Subset(clean_data, assigned_indices)

    # Debug: Print number of images per process
    num_clean_per_process = len(assigned_indices)
    print(f"Process {rank}: Number of clean images assigned: {num_clean_per_process}")
    print(f"Process {rank}: Number of target images: {len(target_data)}")

    # Create DataLoaders
    data_loader_clean = DataLoader(
        clean_subset,
        batch_size=args.batch_size,
        shuffle=False,  # Do not shuffle to ensure unique assignment
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )
    data_loader_target = DataLoader(
        target_data,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle to ensure variety
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )

    # Define inverse normalization for saving images
    inverse_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )

    # Create a cycle iterator for target_loader
    target_loader = itertools.cycle(data_loader_target)

    # Actual number of samples per process
    actual_num_samples = len(assigned_indices)  # per process

    # Initialize tqdm progress bar only for rank 0
    if rank == 0:
        pbar = tqdm(total=actual_num_samples, desc=f"Process {rank} - Processing batches")
    else:
        pbar = None

    if rank == 0:
        print(f"Process {rank}: Actual number of samples to process: {actual_num_samples}")

    for i, (image_org, paths_org) in enumerate(data_loader_clean):
        if image_org.size(0) == 0:
            continue  # Skip empty batches

        # Get target images
        try:
            image_tgt, _ = next(target_loader)
        except StopIteration:
            # Should not happen due to itertools.cycle
            print(f"Process {rank}: Target loader exhausted unexpectedly.")
            break

        # Current batch size
        current_batch_size = image_org.size(0)

        # Move to device
        image_org = image_org.to(device).to(torch.float32)
        image_tgt = image_tgt.to(device).to(torch.float32)
        paths_org = paths_org[:current_batch_size]

        # Extract target image features
        with torch.no_grad():
            tgt_image_embeds = model.ln_vision(model.visual_encoder(image_tgt))
            tgt_image_features = tgt_image_embeds / tgt_image_embeds.norm(dim=-1, keepdim=True)

        # Initialize perturbation
        delta = torch.zeros_like(image_org, requires_grad=True).to(device)

        # Adversarial attack steps
        for step in range(args.steps):
            # Calculate adversarial image
            adv_image = image_org + delta
            adv_image = torch.clamp(adv_image, -1.0, 1.0)  # Keep within [-1, 1]

            # Forward pass
            adv_image_embeds = model.ln_vision(model.visual_encoder(adv_image))
            adv_image_features = adv_image_embeds / adv_image_embeds.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(adv_image_features, tgt_image_features, dim=-1)
            embedding_sim = cosine_sim.mean()

            # Loss and backpropagation
            loss = embedding_sim
            loss.backward()

            # Check if gradients are available
            if delta.grad is None:
                print(f"Process {rank}: Warning: delta.grad is None. Skipping update.")
                break

            # Update perturbation
            with torch.no_grad():
                grad_sign = delta.grad.sign()
                delta.add_(alpha * grad_sign)
                delta.clamp_(-epsilon, epsilon)
                # Keep adversarial image within [-1, 1]
                delta.data = (image_org + delta).clamp(-1.0, 1.0) - image_org

            # Zero gradients
            delta.grad.zero_()

            # Print progress every 50 steps for rank 0, and every 100 steps for others
            if (step + 1) % 50 == 0 or step == args.steps -1:
                if rank == 0 or (rank != 0 and (step + 1) % 100 == 0):
                    print(f"Process {rank} - Step {step+1}/{args.steps}, Cosine Similarity={embedding_sim.item():.5f}")

        # Generate adversarial image
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image, -1.0, 1.0)

        # Inverse normalization and save images
        adv_image_de = inverse_normalize(adv_image)
        adv_image_de = torch.clamp(adv_image_de, 0.0, 1.0)

        # Create output directory if not exists
        process_output_dir = os.path.join(args.output, f"rank_{rank}")
        os.makedirs(process_output_dir, exist_ok=True)

        # Save adversarial images to process-specific subdirectory
        for idx in range(len(paths_org)):
            name = os.path.basename(paths_org[idx])
            save_path = os.path.join(process_output_dir, name)
            torchvision.utils.save_image(adv_image_de[idx].to(torch.float32), save_path)

        # Update progress bar
        if pbar is not None:
            pbar.update(current_batch_size)

    if pbar is not None:
        pbar.close()

    # Synchronize all processes to ensure all have finished processing
    dist.barrier()

    # Only the master process (rank 0) will perform the file moving and directory deletion
    if rank == 0:
        print("All processes have completed. Moving adversarial images to the main output directory and deleting subdirectories.")

        # Define the main output directory
        main_output_dir = args.output
        os.makedirs(main_output_dir, exist_ok=True)

        # Iterate through each rank_* subdirectory
        for rank_dir in os.listdir(main_output_dir):
            rank_subdir = os.path.join(main_output_dir, rank_dir)
            if os.path.isdir(rank_subdir) and rank_dir.startswith("rank_"):
                # Iterate through each file in the subdirectory
                for filename in os.listdir(rank_subdir):
                    src_path = os.path.join(rank_subdir, filename)
                    dest_path = os.path.join(main_output_dir, filename)

                    # Check if the destination file already exists
                    if os.path.exists(dest_path):
                        print(f"Warning: File {dest_path} already exists. Overwriting.")
                        # Optionally, you can raise an error or skip the file
                        # raise ValueError(f"File {dest_path} already exists.")

                    # Move the file
                    shutil.move(src_path, dest_path)
                    # print(f"Moved {src_path} to {dest_path}")

                # After moving all files, delete the empty subdirectory
                try:
                    os.rmdir(rank_subdir)
                    # print(f"Deleted directory: {rank_subdir}")
                except OSError as e:
                    print(f"Error deleting directory {rank_subdir}: {e}")

        print("All adversarial images have been consolidated into the main output directory.")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
