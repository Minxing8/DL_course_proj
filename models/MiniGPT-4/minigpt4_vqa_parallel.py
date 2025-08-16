import argparse
import os
import random
import csv
import logging
import time

from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import torch.backends.cudnn as cudnn
import itertools
import shutil

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

# Seed for reproducibility
DEFAULT_RANDOM_SEED = 2024

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # Not needed since each process uses one GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

        return image_processed, path, index  # Return index for ordering

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
    seedEverything()
    parser = argparse.ArgumentParser(description="MiniGPT-4 Multi-Question Image-to-Text with Multi-GPU")

    # MiniGPT-4 configurations
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model. (Unused in DDP)")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecated), "
             "use --cfg-options instead.",
    )

    # Image and query parameters
    parser.add_argument("--img_path", required=True, type=str, help="Path to the images")
    parser.add_argument("--queries", nargs='+', default=['What is the content of this image?'], type=str, help="List of questions to ask the model")

    # Output parameters
    parser.add_argument("--output_dir", default="../_output_text", type=str, help="Directory to save the output")
    parser.add_argument("--output_name", default="results", type=str, help="Base name for the output files")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for DataLoader")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to process")
    args = parser.parse_args()

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging (each process logs to its own file)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, f'processing_rank_{rank}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Starting processing on rank {rank}...")

    if rank == 0:
        print(f"Loading MiniGPT-4 model...")
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = local_rank
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    chat = Chat(model, vis_processor, device=device)
    if rank == 0:
        print("Model loaded successfully.")
        logging.info("Model loaded successfully.")

    # Load images
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)

    # Determine total number of samples
    total_samples = min(args.num_samples, len(imagenet_data))
    if rank == 0:
        logging.info(f"Total number of samples to process: {total_samples}")
        print(f"Total number of samples to process: {total_samples}")

    # Partition indices uniquely for each process
    assigned_indices = partition_indices(total_samples, world_size, rank)
    imagenet_subset = Subset(imagenet_data, assigned_indices)

    num_samples_per_process = len(assigned_indices)
    logging.info(f"Process {rank}: Number of images assigned: {num_samples_per_process}")
    print(f"Process {rank}: Number of images assigned: {num_samples_per_process}")

    # Create DataLoader
    dataloader = DataLoader(
        imagenet_subset,
        batch_size=args.batch_size,
        shuffle=False,  # Do not shuffle to maintain order
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )

    # Process each question
    for q_idx, question in enumerate(args.queries):
        logging.info(f"Processing Question {q_idx+1}: {question}")
        print(f"Process {rank}: Processing Question {q_idx+1}: {question}")

        # Prepare output CSV file per process and per question
        output_file = os.path.join(
            args.output_dir,
            f"{args.output_name}_question_{q_idx+1}_rank_{rank}.csv"
        )
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['index', 'filename', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            processed_samples = 0
            for i, (images, paths, indices) in enumerate(tqdm(dataloader, desc=f"Process {rank} - Question {q_idx+1}")):
                if processed_samples >= num_samples_per_process:
                    break  # Reached the assigned number of samples

                batch_size = images.size(0)
                for b in range(batch_size):
                    if processed_samples >= num_samples_per_process:
                        break
                    img = images[b].unsqueeze(0).to(device)  # Single image tensor
                    img_path = paths[b]
                    img_index = indices[b].item()  # Original index in the dataset

                    # Initialize conversation
                    conversation = CONV_VISION_Vicuna0.copy()

                    try:
                        # Encode the image
                        temp_img_list = [img]
                        chat.encode_img(temp_img_list)
                        if not temp_img_list:
                            raise ValueError("No embedding returned from encode_img.")
                        embedding = temp_img_list.pop()
                        if embedding is None:
                            raise ValueError("Embedding is None.")

                        # Upload embedding to conversation
                        chat.upload_img(embedding, conversation, [])

                        # Add the user query to the conversation
                        chat.ask(question, conversation)

                        # Generate answer
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

                    # Write to CSV with index
                    writer.writerow({'index': img_index, 'filename': img_path, 'answer': answer})

                    # Release resources
                    del img, temp_img_list, embedding, conversation
                    torch.cuda.empty_cache()

                    processed_samples += 1

            logging.info(f"Results for question {q_idx+1} are saved to {output_file}")
            print(f"Process {rank}: Results for question {q_idx+1} are saved to {output_file}")

    logging.info(f"Process {rank}: All questions processed.")
    print(f"Process {rank}: All questions processed.")

    # Synchronize all processes
    dist.barrier()

    # Only the master process (rank 0) will perform the merging
    if rank == 0:
        print("All processes have completed. Merging output files...")
        for q_idx, question in enumerate(args.queries):
            merged_output_file = os.path.join(
                args.output_dir,
                f"{args.output_name}_question_{q_idx+1}.csv"
            )
            with open(merged_output_file, 'w', newline='', encoding='utf-8') as outfile:
                fieldnames = ['filename', 'answer']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                # Collect all partial output files
                partial_files = [
                    os.path.join(
                        args.output_dir,
                        f"{args.output_name}_question_{q_idx+1}_rank_{r}.csv"
                    ) for r in range(world_size)
                ]

                # Read all partial files and collect rows
                all_rows = []
                for pf in partial_files:
                    with open(pf, 'r', newline='', encoding='utf-8') as infile:
                        reader = csv.DictReader(infile)
                        for row in reader:
                            all_rows.append({
                                'index': int(row['index']),
                                'filename': row['filename'],
                                'answer': row['answer']
                            })

                # Sort rows by index to maintain original order
                all_rows.sort(key=lambda x: x['index'])

                # Write to merged output file
                for row in all_rows:
                    writer.writerow({'filename': row['filename'], 'answer': row['answer']})

            print(f"Merged results for question {q_idx+1} are saved to {merged_output_file}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
