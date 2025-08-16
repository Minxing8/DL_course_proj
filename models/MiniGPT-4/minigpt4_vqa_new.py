import argparse
import os
import random
import csv
import logging
import time  # <-- added for timing

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
device = "cuda" if torch.cuda.is_available() else "cpu"

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecated), change to --cfg-options instead.",
    )

    # Image and query parameters
    parser.add_argument("--img_path", default='/raid/common/imagenet-raw/val/', type=str, help="Path to the images")
    parser.add_argument("--queries", nargs='+', default=['What is the content of this image?'], type=str, help="List of questions to ask the model")

    # Output parameters
    parser.add_argument("--output_dir", default="../_output_text", type=str, help="Directory to save the output")
    parser.add_argument("--output_name", default="results", type=str, help="Base name for the output files")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for DataLoader")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to process")
    # New arguments for timing and tagging
    parser.add_argument("--dataset_tag", default="vqa_unknown", type=str, help="Tag for the dataset (e.g., car_clean)")
    parser.add_argument("--time_log", default="time.txt", type=str, help="Path to time log file.")

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
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    
    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')     
    print("Model loaded successfully.")
    logging.info("Model loaded successfully.")

    # Load images
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logging.info(f"DataLoader batch size: {dataloader.batch_size}")
    print(f"DataLoader batch size: {dataloader.batch_size}")

    # Determine total number of samples
    total_samples = min(args.num_samples, len(imagenet_data))
    logging.info(f"Number of samples to process: {total_samples}")
    print(f"Number of samples to process: {total_samples}")

    total_processing_time = 0.0  # Accumulate processing time per image
    total_images_processed = 0

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
            for i, (images, paths) in enumerate(tqdm(dataloader, desc=f"Processing Question {q_idx+1}")):
                if processed_samples >= total_samples:
                    break  # Reached the desired number of samples

                batch_size = images.size(0)
                for b in range(batch_size):
                    if processed_samples >= total_samples:
                        break
                    img = images[b].unsqueeze(0).to(device)  # Single image tensor
                    img_path = paths[b]

                    # Record start time for this image processing
                    start_time = time.time()

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

                    # Record end time and accumulate processing time
                    elapsed = time.time() - start_time
                    total_processing_time += elapsed
                    total_images_processed += 1

                    # Write to CSV
                    writer.writerow({'filename': img_path, 'answer': answer})
                    processed_samples += 1

            logging.info(f"Results for question {q_idx+1} are saved to {output_file}")
            print(f"Results for question {q_idx+1} are saved to {output_file}")

    logging.info("All questions processed.")
    print("All questions processed.")

    if total_images_processed > 0:
        avg_time = total_processing_time / total_images_processed
    else:
        avg_time = 0.0
    print(f"Average time per image (captioning): {avg_time:.5f} seconds.")

    # Append timing info to the time log file
    with open(args.time_log, "a") as f:
        f.write(f"minigpt4_on_{args.dataset_tag}: {avg_time:.5f} seconds per image (vqa)\n")

if __name__ == "__main__":
    main()
