import argparse
import os
import random
from PIL import Image
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

# Imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


# Seed for reproducibility
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# Basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# Torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Combine seeds
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  


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

        # Debug: Print processed image shape
        # if isinstance(image_processed, torch.Tensor):
        #     print(f"Processed image tensor shape: {image_processed.shape}")  # Expected: [3, 224, 224]
        # else:
        #     print(f"Processed image type: {type(image_processed)}")

        return image_processed, original_tuple[1], path


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser(description="Demo")
    
    # MiniGPT-4
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    
    # Obtain text in batch
    parser.add_argument("--img_file", default='/raid/common/imagenet-raw/val/n01440764/ILSVRC2012_val_00003014.png', type=str)
    parser.add_argument("--img_path", default='/raid/common/imagenet-raw/val/', type=str)
    parser.add_argument("--query", default='what is the content of this image?', type=str)
    
    parser.add_argument("--output_path", default="clean_test_minigpt4", type=str)
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for DataLoader")
    parser.add_argument("--batch_size_in_gen", default=1, type=int, help="Batch size during generation")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to process")
    args = parser.parse_args()
    
    print(f"Loading MiniGPT-4 model...")
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')

    # Initialize vis_processor is already done above using torchvision.transforms
    num_beams = 1
    temperature = 1.0
    print("Done.")

    # Load images
    imagenet_data = ImageFolderWithPaths(args.img_path, transform=None)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=24)
    print(f"DataLoader batch size: {dataloader.batch_size}")  # Confirm batch_size

    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')     
    
    # Create output directory if it doesn't exist
    os.makedirs("../_output_text", exist_ok=True)

    # img2txt
    for i, (image, _, path) in enumerate(dataloader):
        # Initialize conversation for each image
        conversation = CONV_VISION_Vicuna0.copy()
        
        start = time.perf_counter()
        
        print(f"MiniGPT4 img2txt: {i+1}/{args.num_samples//args.batch_size}")
        if i >= args.num_samples // args.batch_size:
            print(f"Successfully processed {args.num_samples} images to text!")
            break 
        image = image.to(device)
        with torch.no_grad():
            # Encode the image
            temp_img_list = [image]  # [1, 3, 224, 224]
            chat.encode_img(temp_img_list)
            
            if temp_img_list:
                embedding = temp_img_list.pop()
                # print(f"[DEBUG] Embedding shape: {embedding.shape}")  # Expected: [1, 32, 4096]
            else:
                # print("[DEBUG] No embedding returned from encode_img.")
                continue  # Skip to next image
            
            # Upload the embedding to the conversation
            chat.upload_img(embedding, conversation, [])  # Pass empty list to avoid modifying img_list
            
            # Add the user query to the conversation
            chat.ask(args.query, conversation)

            # Generate caption
            try:
                caption, _ = chat.answer(conversation, [embedding])
                # print(f"[DEBUG] Generated caption: {caption}")
            except Exception as e:
                print(f"Error generating caption for image {path}: {e}")
                continue  # Skip to next image

        # Write caption to file
        with open(os.path.join("../_output_text", f"{args.output_path}_pred.txt"), 'a') as f:
            f.write(caption + '\n')
        
        end = time.perf_counter()
        print(f"query time for 1 sample:", (end - start))
        
    print("Captions saved.")
