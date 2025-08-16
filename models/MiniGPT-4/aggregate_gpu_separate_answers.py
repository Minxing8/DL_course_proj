import argparse
import os
import csv

def main():
    parser = argparse.ArgumentParser(description="Merge partial output CSV files from multiple ranks into a single CSV per question.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where partial output files are stored.")
    parser.add_argument("--output_name", required=True, type=str, help="Base name for the output files.")
    parser.add_argument("--num_questions", required=True, type=int, help="Number of questions.")
    parser.add_argument("--world_size", required=True, type=int, help="Number of ranks/processes.")
    args = parser.parse_args()

    for q_idx in range(args.num_questions):
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
                ) for r in range(args.world_size)
            ]

            # Read all partial files and collect rows
            all_rows = []
            for pf in partial_files:
                if not os.path.exists(pf):
                    print(f"Warning: Partial file {pf} does not exist. Skipping.")
                    continue
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

if __name__ == '__main__':
    main()
