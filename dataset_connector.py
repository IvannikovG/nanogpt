from datasets import load_dataset

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("dell-research-harvard/AmericanStories",
                       "subset_years",
                       year_list=["1805",
                                  # '1922',
                                  '1960'])
# Initialize an empty list to hold all articles
all_articles = []

# Loop through each year in the dataset
for year in dataset.keys():
    # Access the dataset for the specific year
    year_dataset = dataset[year]
    # Extract the 'article' column and add to the list
    articles = [row['article'] for row in year_dataset]
    all_articles.extend(articles)

# Concatenate all articles into a single string
full_text = ' '.join(all_articles)

# Process the concatenated text
chars = sorted(list(set(full_text)))

# Save the concatenated text to a .txt file
output_file_path = "concatenated_articles.txt"
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(full_text)

print("Text data has been saved to", output_file_path)

# Optional: Print the number of unique characters and a sample of the text
print("Number of unique characters:", len(chars))
print("Sample text:", full_text[:1000])  # Print the first 1000 characters for verification

