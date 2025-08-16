import os
import pandas as pd

# source paths
dir1 = '/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/cele_adv_minigpt4_byminigpt4_1'
dir2 = '/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/cele_adv_minigpt4_byminigpt4_2'

# output path
output_dir = '/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/cele_adv_minigpt4_byminigpt4'

# make output dir
os.makedirs(output_dir, exist_ok=True)

# files list
file_names = [
    "results_question_1.csv",
    "results_question_2.csv",
    "results_question_3.csv",
    "results_question_4.csv",
    "results_question_5.csv"
]

# traverse the files
for file_name in file_names:
    # read files in the first path
    df1 = pd.read_csv(os.path.join(dir1, file_name))
    
    # read the files in the second path without index
    df2 = pd.read_csv(os.path.join(dir2, file_name), header=0)
    
    # concatenate two DataFrame
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # save the new DataFrame to output path
    output_file_path = os.path.join(output_dir, file_name)
    combined_df.to_csv(output_file_path, index=False)
    
    print(f"Successfully merged and saved {file_name} to {output_file_path}")

print("All files have been successfully merged and saved.")
