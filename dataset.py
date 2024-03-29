import os
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
)

BASE_FILENAMES = [
    "data/{}-1.txt",
    "data/{}-2.txt",
    "data/{}-3.txt",
    "data/{}-4.txt",
    "data/{}-5.txt",
]

INSTRUCTION = "Summarize the following story in my style"

# Initialize empty DataFrames to store instructions, stories, and summaries
df_instructions = pd.DataFrame(columns=['text'])
df_stories = pd.DataFrame(columns=['text'])
df_summaries = pd.DataFrame(columns=['text'])

print(f"{'Total':<12}{'Instruction':<12}{'Story':<12}{'Summary':<12}")
for base_filename in BASE_FILENAMES:
    # Read the story and the summary files
    with open(base_filename.format("story"), 'r') as file:
        story = "".join(file.readlines())
    with open(base_filename.format("summary"), "r") as file:
        summary = "".join(file.readlines())
    
    # Count tokens
    instruction_tokens = tokenizer(INSTRUCTION, return_tensors="pt")["input_ids"].shape[1]
    story_tokens = tokenizer(story, return_tensors="pt")["input_ids"].shape[1]
    summary_tokens = tokenizer(summary, return_tensors="pt")["input_ids"].shape[1]
    
    # Print table of tokens
    total_tokens = instruction_tokens + story_tokens + summary_tokens
    print(f"{total_tokens:<12}{instruction_tokens:<12}{story_tokens:<12}{summary_tokens:<12}")
    
    # Append data to the DataFrames
    df_instructions = pd.concat([df_instructions, pd.DataFrame([{'text': INSTRUCTION}])], ignore_index=True)
    df_stories = pd.concat([df_stories, pd.DataFrame([{'text': story}])], ignore_index=True)
    df_summaries = pd.concat([df_summaries, pd.DataFrame([{'text': summary}])], ignore_index=True)

# Write the DataFrames to Parquet files
df_instructions.to_parquet('instructions.parquet', index=False)
df_stories.to_parquet('stories.parquet', index=False)
df_summaries.to_parquet('summaries.parquet', index=False)
