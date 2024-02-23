import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

# ------------------------------------------
# --- Loading the data and preprocessing ---
# ------------------------------------------

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
def load_data(directory_path):
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Check if the file is a text document
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    sentence, score = line.strip().split('\t')
                    all_data.append((sentence, int(score)))
    return all_data

# Preprocessing steps
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text

# Directory path to the dataset containing text documents
directory_path = '../sentiment labelled sentences'  # Corrected path

# Load all three datasets
all_data = load_data(directory_path)

# Separate sentences and labels
sentences = [data[0] for data in all_data]
scores = [data[1] for data in all_data]

# Apply preprocessing steps
preprocessed_data = [preprocess_text(text) for text in sentences]

# Create a DataFrame
df = pd.DataFrame({'Text': preprocessed_data, 'Score': scores})

# Display the preprocessed data
print(df.head())

# ------------------------------------------
# *** Loading the data and preprocessing ***
# ------------------------------------------

# ------------------------------------------
# --------- Training a classifier ----------
# ------------------------------------------



# ------------------------------------------
# ********* Training a classifier **********
# ------------------------------------------