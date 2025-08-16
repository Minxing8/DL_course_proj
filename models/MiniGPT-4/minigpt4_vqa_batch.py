import argparse
import os
import random
import csv
import logging
import time
import pickle

from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

# Seed for reproducibility
DEFAULT_RANDOM_SEED = 2024
device = f"cuda:{0}" if torch.cuda.is_available() else "cpu"  # Use specific GPU if available

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Use benchmark if input size fixed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Define standard image preprocessing pipeline using torchvision.transforms
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

def main():
    seedEverything()
    parser = argparse.ArgumentParser(description="MiniGPT-4 Multi-Question Image-to-Text")

    # MiniGPT-4 configurations
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecated), "
             "change to --cfg-options instead.",
    )

    # Image and query parameters
    parser.add_argument("--img_path", default='/raid/common/imagenet-raw/val/', type=str, help="Path to the images")
    parser.add_argument("--queries", nargs='+', default=['What is the content of this image?'], type=str, help="List of questions to ask the model")

    # Output parameters
    parser.add_argument("--output_dir", default="../_output_text", type=str, help="Directory to save the output")
    parser.add_argument("--output_name", default="results", type=str, help="Base name for the output files")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for DataLoader and image encoding")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to process")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'processing.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting processing...")

    print(f"Loading MiniGPT-4 model...")
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
    model = model_cls.from_config(model_config).to(device)
    model.eval()  # Set model to evaluation mode
    logging.info("Model loaded and set to evaluation mode.")
    print("Model loaded and set to evaluation mode.")

    # Initialize Chat
    chat = Chat(model, vis_processor, device=device)
    print("Chat initialized.")
    logging.info("Chat initialized.")

    # Optimize DataLoader
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)
    dataloader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,        # Increase num_workers for faster data loading
        pin_memory=True       # Enable pin_memory for faster data transfer to GPU
    )
    logging.info(f"DataLoader batch size: {dataloader.batch_size}, num_workers: 12, pin_memory: True")
    print(f"DataLoader batch size: {dataloader.batch_size}")

    # Determine total number of samples
    total_samples = min(args.num_samples, len(imagenet_data))
    logging.info(f"Number of samples to process: {total_samples}")
    print(f"Number of samples to process: {total_samples}")

    # Pre-encode all images and cache using chat.encode_img
    cache_file = os.path.join(args.output_dir, 'image_features.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            image_features = pickle.load(f)
        logging.info("Loaded cached image features.")
        print("Loaded cached image features.")
    else:
        image_features = {}
        logging.info("Encoding images using chat.encode_img...")
        print("Encoding images using chat.encode_img...")
        for images, paths in tqdm(dataloader, desc="Encoding images"):
            batch_size = images.size(0)
            for b in range(batch_size):
                if len(image_features) >= total_samples:
                    break
                img = images[b].unsqueeze(0).to(device)  # Single image tensor
                img_path = paths[b]

                # Initialize conversation
                conversation = CONV_VISION_Vicuna0.copy()

                try:
                    # Encode the image using chat.encode_img
                    temp_img_list = [img]
                    chat.encode_img(temp_img_list)
                    if not temp_img_list:
                        raise ValueError("No embedding returned from encode_img.")
                    embedding = temp_img_list.pop()
                    if embedding is None:
                        raise ValueError("Embedding is None.")

                    # Store the embedding
                    image_features[img_path] = embedding.cpu()
                except Exception as e:
                    logging.error(f"Error encoding image {img_path}: {e}")
                    image_features[img_path] = None  # Mark as failed

                # Release resources
                del img, temp_img_list, embedding
                torch.cuda.empty_cache()

        # Cache image features
        with open(cache_file, 'wb') as f:
            pickle.dump(image_features, f)
        logging.info("Image features cached.")
        print("Image features cached.")

    # Process each question
    for q_idx, question in enumerate(args.queries):
        logging.info(f"Processing Question {q_idx+1}: {question}")
        print(f"Processing Question {q_idx+1}: {question}")

        # Prepare output CSV file
        output_file = os.path.join(args.output_dir, f"{args.output_name}_question_{q_idx+1}.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            processed_samples = 0
            image_paths_list = [path for path in image_features.keys() if image_features[path] is not None][:total_samples]
            for img_path in tqdm(image_paths_list, desc=f"Processing Question {q_idx+1}"):
                if processed_samples >= total_samples:
                    break

                embedding = image_features[img_path].to(device)

                # Initialize conversation
                conversation = CONV_VISION_Vicuna0.copy()

                try:
                    # Upload image embedding to conversation
                    chat.upload_img(embedding, conversation, [])

                    # Add user question to conversation
                    chat.ask(question, conversation)

                    # Generate answer, adjust generation parameters to speed up
                    answer, _ = chat.answer(
                        conversation,
                        [embedding],
                        max_new_tokens=50,
                        num_beams=1,
                        temperature=1.0,
                        length_penalty=0.5
                    )
                    if not answer:
                        raise ValueError("No answer generated.")
                    answer = answer.replace('\n', ' ')  # Ensure single line
                except Exception as e:
                    logging.error(f"Error generating answer for image {img_path}: {e}")
                    answer = "Error"

                # Write to CSV
                writer.writerow({'filename': img_path, 'answer': answer})

                # Release resources
                del embedding, conversation
                torch.cuda.empty_cache()

                processed_samples += 1

        logging.info(f"Results for question {q_idx+1} are saved to {output_file}")
        print(f"Results for question {q_idx+1} are saved to {output_file}")

    logging.info("All questions processed.")
    print("All questions processed.")

if __name__ == "__main__":
    main()
