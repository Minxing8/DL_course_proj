import argparse
import os
import csv
import torch
from PIL import Image
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_llava_v1
from llava.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def main():
    parser = argparse.ArgumentParser(description="LLAVA Image-to-Text VQA Processing")
    parser.add_argument("--model_name_or_path", default="liuhaotian/llava-v1.6-mistral-7b", help="Model name or path")
    parser.add_argument("--model_base", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base language model path")
    parser.add_argument("--image_dir", required=True, help="Directory of images")
    parser.add_argument("--output_dir", default="llava_output", help="Directory to save the outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--questions", nargs="+", default=["Describe the image."], help="List of questions to ask")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of images to process")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top p for sampling")
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for beam search")

    args = parser.parse_args()

    is_lora = args.model_base is not None and "lora" in args.model_name_or_path.lower()
    # -------------------------------------------------------------
    # Select ONE conversation template and store it in conv_template
    # -------------------------------------------------------------
    # if is_lora:
    #     from llava.conversation import conv_llava_plain as conv_template
    # else:
    #     from llava.conversation import conv_llava_v1    as conv_template

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model_base in [None, "None"]:
        args.model_base = None

    # Load model and processor
    try:
        # tokenizer, model, image_processor, context_len = load_pretrained_model(
        #     model_path=args.model_name_or_path,
        #     model_base=args.model_base,
        #     model_name=get_model_name_from_path(args.model_name_or_path),
        #     torch_dtype=torch.float16,
        # )


        if args.model_base is not None and "lora" in args.model_name_or_path.lower():
            name_for_builder = get_model_name_from_path(args.model_name_or_path)
        else:
            name_for_builder = get_model_name_from_path(args.model_base or args.model_name_or_path)


        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_name_or_path,
            model_base=args.model_base,
            model_name=name_for_builder,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # ————————————————————————————————————————
        # If we're in LoRA mode, builder.py never sets up vision_tower or image_processor,
        # and it never adds the special image tokens to the tokenizer.  Do it here:
        from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        # 1) Add image tokens exactly as builder.py does for pure-"llava" runs:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        # 2) If image_processor is None, manually load the CLIP vision tower:
        if image_processor is None:
            # get the CLIP wrapper
            vision_tower = model.get_vision_tower()
            # load if needed
            if not vision_tower.is_loaded:
                vision_tower.load_model(device_map="auto")
            # move to GPU + half
            vision_tower.to(device=model.device, dtype=torch.float16)
            # grab its processor
            image_processor = vision_tower.image_processor
        # ————————————————————————————————————————

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Add special tokens to tokenizer
    if "llava" in args.model_name_or_path.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    # Verify vision_tower is loaded and on device
    # vision_tower = model.get_vision_tower()
    # if vision_tower is None:
    #     raise ValueError("Vision tower is not loaded correctly. Please check the model path and model_base.")
    # else:
    #     print("Vision tower loaded successfully.")
    #     vision_tower.to(device)

    # Get list of image paths
    image_paths = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png")):
                image_paths.append(os.path.join(root, file))

    total_images = min(len(image_paths), args.num_samples)
    image_paths = image_paths[:total_images]
    print(f"Total images to process: {total_images}")

    # Prepare logging for empty answers
    empty_answers_log = os.path.join(args.output_dir, "empty_answers.log")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(empty_answers_log, "w", encoding="utf-8") as log_file:
        log_file.write("Empty Answers Log\n")
        log_file.write("=================\n")

    # Process each question
    for question_idx, question in enumerate(args.questions):
        print(f"Processing Question {question_idx + 1}: {question}")

        # Create output directory for the question
        question_output_dir = os.path.join(
            args.output_dir, f"question_{question_idx + 1}"
        )
        os.makedirs(question_output_dir, exist_ok=True)

        # Create output CSV file
        output_file = os.path.join(question_output_dir, "results.csv")

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "answer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Batch processing images
            for i in tqdm(range(0, total_images, args.batch_size), desc=f"Processing Question {question_idx + 1}"):
                batch_paths = image_paths[i : i + args.batch_size]
                images = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading image {p}: {e}")
                        images.append(None)

                # Preprocess images
                image_tensors = []
                image_sizes = []
                for img in images:
                    if img is not None:
                        try:
                            processed_image = image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)
                            image_tensors.append(processed_image)
                            image_sizes.append(list(img.size))  # Convert tuple to list
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            # Use a default tensor if processing fails
                            image_tensors.append(torch.zeros((1, 3, 336, 336)).to(device, dtype=torch.float16))
                            image_sizes.append([336, 336])
                    else:
                        # Use a default tensor if image is None
                        image_tensors.append(torch.zeros((1, 3, 336, 336)).to(device, dtype=torch.float16))
                        image_sizes.append([336, 336])

                # Stack image tensors into batch
                image_tensors = torch.cat(image_tensors, dim=0)
                # print(f"Image tensors shape: {image_tensors.shape}")

                # Build prompts with conversation template
                prompts = []
                for _ in images:
                    # conversation = conv_llava_v1.copy()
                    # for pure pretrained VQA we keep the chat template;
                    # for our OCR‐LoRA runs, switch to the simpler “plain” template
                    #------------------------------------------------------
                    if is_lora:
                        from llava.conversation import conv_llava_plain as _conv_template
                        conversation = _conv_template.copy()
                        # patch sep2 so it's also a newline (instead of None)
                        conversation.sep2 = conversation.sep
                    else:
                        conversation = conv_llava_v1.copy()
                    # conversation = conv_template.copy()
                    # if is_lora:                       # keep the sep2 patch for plain template
                    #     conversation.sep2 = conversation.sep
                    #------------------------------------------------------

                    if getattr(model.config, "mm_use_im_start_end", False):
                        image_placeholder = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
                    else:
                        image_placeholder = DEFAULT_IMAGE_TOKEN
                    conversation.append_message(conversation.roles[0], image_placeholder)
                    conversation.append_message(conversation.roles[0], question)
                    prompt = conversation.get_prompt()
                    prompts.append(prompt)

                # Tokenize prompts
                input_ids_list = []
                for prompt in prompts:
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",)
                    input_ids_list.append(input_ids.squeeze(0))
                # Pad sequences
                input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.eos_token_id)
                # Move tensors to device
                input_ids_padded = input_ids_padded.to(device)
                # print(f"Input IDs shape: {input_ids_padded.shape}")

                # Prepare generate parameters
                # generate_kwargs = dict(
                #     max_new_tokens=args.max_new_tokens,
                #     temperature=args.temperature,
                #     top_p=args.top_p,
                #     num_beams=args.num_beams,
                #     do_sample=True if args.temperature > 0 else False,
                #     # do_sample=False,
                #     use_cache=True,
                #     pad_token_id=tokenizer.eos_token_id,
                # )
                #------------------------------------------------------
                if is_lora:
                    # OCR‐style deterministic beams for LoRA
                    generate_kwargs = dict(
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        num_beams=args.num_beams,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    # original pretrained sampling settings
                    generate_kwargs = dict(
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        do_sample=True if args.temperature > 0 else False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                #------------------------------------------------------

                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids_padded,  # Position arguments
                            images=image_tensors,
                            image_sizes=image_sizes,
                            **generate_kwargs,
                        )
                    # Decode outputs
                    answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    # Extract assistant's reply
                    for idx in range(len(answers)):
                        output = answers[idx]
                        # Robust parsing
                        delimiter = conv_llava_v1.roles[1] + ":"
                        # delimiter = conv_template.roles[1] + ":"
                        if delimiter in output:
                            answer = output.split(delimiter)[-1].strip()
                        else:
                            # If delimiter not found, check for leading colon or use entire output
                            if output.startswith(":"):
                                answer = output[1:].strip()
                            else:
                                answer = output.strip()
                        # Handle empty answers
                        if not answer:
                            answer = "No answer generated."
                            # Log the empty answer
                            with open(empty_answers_log, "a", encoding="utf-8") as log_file:
                                log_file.write(f"Empty answer for image: {batch_paths[idx]}\n")
                        answers[idx] = answer
                except Exception as e:
                    print(f"Error during generation: {e}")
                    answers = ["Error" for _ in batch_paths]
                    # Log the error
                    with open(empty_answers_log, "a", encoding="utf-8") as log_file:
                        for img_path in batch_paths:
                            log_file.write(f"Generation error for image: {img_path}\n")

                # Write results to CSV
                for img_path, answer in zip(batch_paths, answers):
                    answer = answer.replace('\n', ' ')
                    writer.writerow({"filename": img_path, "answer": answer})

        print(f"Results for question {question_idx + 1} are saved to {output_file}")

    print("All questions processed.")


if __name__ == "__main__":
    main()