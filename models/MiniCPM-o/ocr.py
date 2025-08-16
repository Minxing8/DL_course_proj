# minicpmo_ocr.py

import argparse
import os
import csv
import traceback
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o-2.6 OCR/VQA on custom image set")
    parser.add_argument("--model_name_or_path", default="openbmb/MiniCPM-o-2_6",
                        help="MiniCPM-o checkpoint on Hugging Face")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", default="minicpmo_output", help="Where to write CSV results")
    parser.add_argument("--batch_size", type=int, default=4, help="How many images to process at once")
    parser.add_argument("--questions", nargs="+",
                        default=["What text is in the image?"],
                        help="List of OCR-style questions to ask")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max number of images to process (default = all)")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Max tokens to generate per query")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model + tokenizer
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    # Collect image file paths
    image_paths = []
    for root, _, files in os.walk(args.image_dir):
        for fn in files:
            if fn.lower().endswith(("jpg", "jpeg", "png")):
                image_paths.append(os.path.join(root, fn))
    if args.num_samples:
        image_paths = image_paths[: args.num_samples]
    print(f"Found {len(image_paths)} images.")

    # Prepare output directories & log
    os.makedirs(args.output_dir, exist_ok=True)
    empty_log = os.path.join(args.output_dir, "empty_answers.log")
    with open(empty_log, "w", encoding="utf-8") as elog:
        elog.write("Empty / Error Answers Log\n=========================\n")

    # Loop over each question
    for qi, question in enumerate(args.questions, start=1):
        print(f"\n=== Question {qi}: {question!r} ===")
        q_out_dir = os.path.join(args.output_dir, f"question_{qi}")
        os.makedirs(q_out_dir, exist_ok=True)
        csv_path = os.path.join(q_out_dir, "results.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["filename", "answer"])
            writer.writeheader()

            # Process in batches
            for i in tqdm(range(0, len(image_paths), args.batch_size), desc=f"Q{qi}"):
                batch = image_paths[i : i + args.batch_size]
                for img_path in batch:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        msgs = [{"role": "user", "content": [img, question]}]

                        # Debug print
                        print(f"[DEBUG] Chat on {img_path}")

                        with torch.no_grad():
                            resp = model.chat(
                                msgs=msgs,
                                tokenizer=tokenizer,
                                sampling=(args.temperature > 0),
                                max_new_tokens=args.max_new_tokens,
                                use_tts_template=False,
                                generate_audio=False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                            )
                        answer = resp.strip()
                        if not answer:
                            raise ValueError("empty response")

                    except Exception as e:
                        # log full traceback
                        tb = traceback.format_exc()
                        with open(empty_log, "a", encoding="utf-8") as elog:
                            elog.write(f"Image: {img_path}\n{tb}\n")
                        answer = f"Error: {e}"

                    writer.writerow({"filename": img_path, "answer": answer})

        print(f"Saved results for question {qi} to {csv_path}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
