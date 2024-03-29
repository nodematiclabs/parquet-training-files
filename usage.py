from datasets import Dataset, load_dataset

# Paths to your Parquet files
instructions_path = 'instructions.parquet'
stories_path = 'stories.parquet'
summaries_path = 'summaries.parquet'

# Loading the datasets
instructions_dataset = load_dataset('parquet', data_files=instructions_path, split='train')
stories_dataset = load_dataset('parquet', data_files=stories_path, split='train')
summaries_dataset = load_dataset('parquet', data_files=summaries_path, split='train')

# Optionally, you can print the first few entries to verify
# print(instructions_dataset[0])
# print(stories_dataset[0])
# print(summaries_dataset[0])

def combine_texts(instruction, story, summary):
    return {
        "text": f"""
{instruction['text']}

### Story
{story['text']}

### Summary
{summary['text']}
"""}

# Assuming the datasets are aligned and have the same number of examples
combined_texts = [combine_texts(instructions_dataset[i], stories_dataset[i], summaries_dataset[i]) for i in range(len(instructions_dataset))]

# Creating a new dataset from the combined texts
final_dataset = Dataset.from_dict({"text": [ct["text"] for ct in combined_texts]})

# Showing an example from the combined dataset
print(final_dataset[0]['text'])
