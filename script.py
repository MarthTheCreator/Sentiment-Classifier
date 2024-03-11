# Import preprocessing libraries
import os
import pandas as pd
import contractions
import unicodedata
import re

# Import NLTK libraries
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import chunk

# >------------------------------------------<
# >-- Functions and controls of the script --<
# >------------------------------------------<

# Lemmatization and stemming controls
use_lemmatization = True
use_stemming = False

# TF-IDF control
use_tfidf = True

# N-gram range control
ngram_range = (1,1)  #(1,1) for unigrams, (1,2) for unigrams and bigrams, or (2,2) for bigrams and so forth. 

# Load the data sets
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

# Function for data cleaning and basic preprocessing. 
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()

    # Remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Expand contractions
    text = contractions.fix(text)

    # Special characters, punctuation, and numbers removal
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # POS Tagging (to be used later in lemmatization if needed)
    pos_tags = pos_tag(tokens)
    
    # In preprocess_text function, after POS tagging:
    return tokens, [tag for _, tag in pos_tags]

# Function for lemmatizing the corpus (with pos-tagging)
def lemmatize_text(tokens, pos_tags):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in zip(tokens, pos_tags)]
    return ' '.join(lemmatized_tokens)

# Function for stemming the corpus
def stem_text(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# Convert NLTK's POS tags to ones used by the WordNet lemmatizer.
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:  # Default case
        return wordnet.NOUN

# >------------------------------------------<
# >** Functions and controls of the script **<
# >------------------------------------------<

# >------------------------------------------<
# >--- Loading the data and preprocessing ---<
# >------------------------------------------<

# Load all three datasets
directory_path = '../sentiment labelled sentences'
all_data = load_data(directory_path)

# Apply preprocessing and decide whether to lemmatize, stem or neither
processed_texts = []
for text, _ in all_data:
    tokens, pos_tags = preprocess_text(text)
    if use_lemmatization:
        processed_text = lemmatize_text(tokens, pos_tags) # Applying lemmatization with postagging
    elif use_stemming:
        processed_text = stem_text(tokens) # Pos-tagging is not needed for applying stemming 
    else:
        processed_text = ' '.join(tokens)  # Neither lemmatization nor stemming applied.
    processed_texts.append(processed_text)

# Prepare the dataset
df = pd.DataFrame({'Text': processed_texts, 'Score': [score for _, score in all_data]})

# >------------------------------------------<
# >*** Loading the data and preprocessing ***<
# >------------------------------------------<

# >------------------------------------------<
# >--------- Training a classifier ----------<
# >------------------------------------------<

# Import models and train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Import vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import evaluation 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(df['Text'], df['Score'], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Text vectorization and n_grams depending on the control settings
if use_tfidf:
    vectorizer = TfidfVectorizer(ngram_range=ngram_range) 
else:
    vectorizer = CountVectorizer(ngram_range=ngram_range)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Model training and predictions
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(kernel='linear'),
    "Gradient Boosting Machine": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

'''
# Loop through models, fit, predict, and print evaluation metrics
for name, model in models.items():
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_val_vectorized)
    print(f"{name} Classifier:")
    print(classification_report(y_val, y_pred))
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\n")
'''

# >------------------------------------------<
# >********* Training a classifier **********<
# >------------------------------------------<


# >------------------------------------------<
# >--------- Hyperparameter tuning ----------<
# >------------------------------------------<

# Hyperparameter grids
param_grids = {
    "Random Forest": {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "Naive Bayes": {
        'alpha': [0.01, 0.1, 1, 10]
    },
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    },
    "Gradient Boosting Machine": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
}

best_models = {}
best_params = {}
best_scores = {}
# Looping through the models with GridSearch
for name, model in models.items():
    print(f"Starting Grid Search for {name}")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_vectorized, y_train)  
    best_models[name] = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_
    best_scores[name] = grid_search.best_score_
    print(f"Best parameters for {name}: {best_params[name]}")
    print(f"Best cross-validation score for {name}: {best_scores[name]:.3f}")
    print("\n")

# >------------------------------------------<
# >********* Hyperparameter tuning **********<
# >------------------------------------------<