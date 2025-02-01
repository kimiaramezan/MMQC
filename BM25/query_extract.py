import os

from openai import OpenAI
import json
import pandas as pd
client = OpenAI()
import json
from openai import OpenAI
from tqdm import tqdm

# Load the dataset

# Load the dataset
file_path = "../Data/sorted_final4_4o_mini.csv"  # Change to your actual dataset path
data = pd.read_csv(file_path)
output_file = "intent_results_4_full.csv"

# Extract relevant columns
columns_to_keep = ["facet", "facet_id", "topic_id", "topic", "image1_1", "image1_2", "image1_3", "image2_1", "image2_2", "image2_3", "image3_1", "image3_2", "image3_3", "image4_1", "image4_2", "image4_3"]
result_data = data[columns_to_keep].copy()

# Extract conversations
queries = []
for _, row in data.iterrows():
    conversation = ""
    for i in range(1, 5):  # Assuming max of 4 questions/answers per entry
        question_key = f"question{i}"
        answer_key = f"answer{i}"
        if pd.notna(row[question_key]) and pd.notna(row[answer_key]):
            conversation += row[question_key] + " " + row[answer_key] + " "
    queries.append(conversation.strip())

# Initialize OpenAI client
client = OpenAI()

# Ensure the DataFrame has a "query" column
result_data["query"] = ""
i = 0
# Process each conversation and update the DataFrame inside the loop
for idx, convo in tqdm(enumerate(queries), desc="Processing Conversations", total=len(queries)):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract the user's intent based on the conversation. DO NOT mention what they are NOT interested in. DO NOT print 'the user intends to' and just print the intent."},
            {"role": "user", "content": convo}
        ]
    )
    intent = completion.choices[0].message.content.strip()

    # Update the specific row in the DataFrame
    result_data.at[idx, "query"] = intent 
    if i < 10:
        print(intent)
        i += 1 

    # Save results to file after each update
    result_data.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")