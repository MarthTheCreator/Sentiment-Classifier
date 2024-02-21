import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
def load_data(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            data.append(text)
    return data

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
directory_path = 'path/to/dataset/'

# Load the data
data = load_data(directory_path)

# Apply preprocessing steps
preprocessed_data = [preprocess_text(text) for text in data]

# Create a DataFrame
df = pd.DataFrame({'Text': preprocessed_data})

# Display the preprocessed data
print(df.head())
