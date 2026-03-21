import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Global Variable
FILE_PATH = 'Part_A_Dataset\Data_1.txt'

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

def nlp_pipeline():
    # Load the dataset using the global variable
    try:
        with open(FILE_PATH, 'r') as f:
            text = f.read()
            # Added to print the original raw text for documentation
            print("\n--- Original Text in Data_1 ---")
            print(text)
            print("-" * 30 + "\n")
    except FileNotFoundError:
        print(f"Error: {FILE_PATH} not found. Please ensure the file is in the correct directory.")
        return

    print("--- Word Tokenization ---")

    # Python split() implementation
    # Simple whitespace-based segmentation
    split_tokens = text.split()
    print("\nTokenisation using split function ~~~~~~~~~~~~~~~")
    print(split_tokens)

    # Regular Expression (re) implementation
    # Pattern-based segmentation to capture alphanumeric characters only
    regex_pattern = r'\w+'
    regex_tokens = re.findall(regex_pattern, text)
    print("\nTokenisation using Regular Expression ~~~~~~~~~~~~~~~")
    print(regex_tokens)

    # NLTK Library implementation
    # Sophisticated linguistic segmentation using pre-trained models
    nltk_tokens = word_tokenize(text)
    print("\nTokenisation using nltk word tokenise function ~~~~~~~~~~~~~~~")
    print(nltk_tokens)


    print("\n\n--- Stop Words and Punctuation Filtering ---")

    # Filtering and Identification
    # Convert to lowercase for consistent matching
    text_lower = text.lower()
    tokenized_lower = word_tokenize(text_lower)
    
    # Define noise: standard English stopwords + string punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    noise_list = stop_words.union(punctuation)
    
    # Identify stop words present in the specific text
    identified_stop_words = sorted(list(set([w for w in tokenized_lower if w in stop_words])))
    
    # Filtered Output
    filtered_tokens = [w for w in tokenized_lower if w not in noise_list]
    
    print(f"\nList of Identified Stop Words in {FILE_PATH}:")
    print(identified_stop_words)
    
    print("\nFiltered Output (Noise Removed) ~~~~~~~~~~~~~~~")
    print(filtered_tokens)
    
    print(f"\nSummary: Original ({len(nltk_tokens)} tokens) -> Filtered ({len(filtered_tokens)} tokens)")

if __name__ == "__main__":
    nlp_pipeline()