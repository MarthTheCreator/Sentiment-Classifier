# Import preprocessing libraries
import os
import pandas as pd
import numpy as np
import contractions
import unicodedata
import re

# Import NLTK libraries
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, chunk

# Import sklkearn libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import pipeline for experiments
from sklearn.pipeline import Pipeline

# Import gensim libraries
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# import SpaCy and Yake libraries
import spacy
from spacy.matcher import Matcher
import yake

# >-----------------------------------------------------------------------<
# >----------------- Loading the data and preprocessing ------------------<
# >-----------------------------------------------------------------------<

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
    
    # Word tokenization
    tokens = word_tokenize(text)
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

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
    else:  
        return wordnet.NOUN

# Function to apply stemming
def stem_preprocessor(text):
    stemmer = PorterStemmer()
    # Tokenize the text, stem each token, then rejoin back to string
    return ' '.join([stemmer.stem(token) for token in word_tokenize(text)])

# Function to apply lemmatization
def lemma_preprocessor(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the text, POS-tag, lemmatize each token, then rejoin back to string
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tag(tokens)])

# No preprocessing function
def no_preprocessor(text):
    return text

# Load all three datasets
directory_path = '../sentiment labelled sentences'
all_data = load_data(directory_path)

# Applying the preprocess_text function to each text in all_data
processed_texts = [' '.join(preprocess_text(text)) for text, _ in all_data]

# Prepare the dataset
df = pd.DataFrame({'Text': processed_texts, 'Score': [score for _, score in all_data]})

# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >------------------------ Training a classifier ------------------------<
# >-----------------------------------------------------------------------<

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Score'], test_size=0.2, random_state=42)

# Vectorizing the data for basic model building. 
model_vectorizer = TfidfVectorizer(use_idf=False)
X_train_vectorized = model_vectorizer.fit_transform(X_train)
X_test_vectorized = model_vectorizer.transform(X_test)

# Model training and predictions
models = {
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting Machine": GradientBoostingClassifier()
}

# Loop through models, fit, predict, and print evaluation metrics
for name, model in models.items():
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)  
    print(f"{name} Classifier:")
    print(classification_report(y_test, y_pred)) 
    print("Accuracy:", accuracy_score(y_test, y_pred)) 
    print("\n")

# basic hyperparametertuning grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

# Initialize Logistic Regression
lr_model = LogisticRegression()

# Initialize GridSearchCV
grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search_lr.fit(X_train_vectorized, y_train)

# Best model, parameters, and score
best_model_lr = grid_search_lr.best_estimator_
best_params_lr = grid_search_lr.best_params_
best_score_lr = grid_search_lr.best_score_

# Output best parameters and score
print(f"Best parameters for Logistic Regression: {best_params_lr}")
print(f"Best cross-validation score for Logistic Regression: {best_score_lr:.3f}")

# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >--------------------------- Experiment 1-3 ----------------------------<
# 1. Stemming, lemmatization or neither
# 2. With tf.idf weigths or without tf.idf weights
# 3. Using complete words or n_grams 
# >-----------------------------------------------------------------------<

# Base model with some hyperparameter tuning 
lr = LogisticRegression(C=10, solver = 'liblinear')

# Define the parameter grid for n-gram ranges and vectorization strategies
# N_gram range for experiment
ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]

# Preprocessing configs: 
# With stemming, lemmatization or neither
# With or without tf.idf weights
preprocessing_configs = [
    ('None', no_preprocessor, 'use_idf=True'),
    ('Stemming', stem_preprocessor, 'use_idf=True'),
    ('Lemmatization', lemma_preprocessor, 'use_idf=True'),
    ('None', no_preprocessor, 'use_idf=False'),
    ('Stemming', stem_preprocessor, 'use_idf=False'),
    ('Lemmatization', lemma_preprocessor, 'use_idf=False')
]

# Store results
experiment_results = []

# Loop through each preprocessing and TF-IDF configuration
for preproc_label, preprocessor_func, tfidf_param in preprocessing_configs:
    # Selecting vectorizer based on TF-IDF parameter
    if 'True' in tfidf_param:
        vect = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, preprocessor=preprocessor_func)
    else:
        vect = TfidfVectorizer(use_idf=False, min_df=0.0, max_df=1.0, preprocessor=preprocessor_func)

    for ngram_range in ngram_ranges:
        # Update vectorizer n-gram range
        vect.set_params(ngram_range=ngram_range)
        
        # Fit and transform the training data with the vectorizer
        X_train_vectorized = vect.fit_transform(X_train)
        X_test_vectorized = vect.transform(X_test)  # Changed from X_val to X_test
        
        # Train and evaluate the Logistic Regression model
        lr.fit(X_train_vectorized, y_train)
        cv_scores = cross_val_score(lr, X_train_vectorized, y_train, cv=5)
        cv_mean_score = np.mean(cv_scores)
        cv_std_dev = np.std(cv_scores)
        test_score = lr.score(X_test_vectorized, y_test) 

        # Store the performance metrics
        experiment_results.append({
            'Preprocessing': preproc_label,
            'TF-IDF': tfidf_param,
            'N-gram range': ngram_range,
            'CV score (std)': cv_std_dev,
            'Mean CV accuracy': cv_mean_score,
        })     

# Print summary of results for all configurations
for result in experiment_results:
    print(f"{result['Preprocessing']} {result['TF-IDF']} {result['N-gram range']} - "
          f"Mean CV accuracy: {result['Mean CV accuracy']}")

# Convert the experiment results to a DataFrame
results_df = pd.DataFrame(experiment_results)

# Add a column for ranking based on 'Mean CV accuracy'
results_df['Rank'] = results_df['Mean CV accuracy'].rank(method='max', ascending=False)

# Sort the DataFrame based on the rank
results_df = results_df.sort_values(by=['Rank'], ascending=True)

# Display the sorted DataFrame
print(results_df)

# Save the results of the experiments as a csv
results_df.to_csv('../experiment1-3_results.csv', index=True)

# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >----- Experiment 4 - BOW representation vs using word embeddings  -----<
# >-----------------------------------------------------------------------<

# Creating Word2Vec models: CBOW and Skip-gram
cbow_model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, sg=0)  # CBOW model
skipgram_model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, sg=1)  # Skip-gram model

def average_word_vectors(words, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0
    for word in words:
        if word in model.wv.index_to_key:
            nwords += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

# Generate the average word vectors for the training sets for both models
X_train_cbow_vec = np.array([average_word_vectors(text, cbow_model, 100) for text in X_train])
X_train_skipgram_vec = np.array([average_word_vectors(text, skipgram_model, 100) for text in X_train])

# Using Support Vector Machine with the best parameters found from hyperparameter tuning
lr_best = LogisticRegression(C=10, solver = 'liblinear')

# Evaluating CBOW model with 5-fold cross-validation
cv_scores_cbow = cross_val_score(lr_best, X_train_cbow_vec, y_train, cv=5, scoring='accuracy')
print("CBOW model - LR 5-fold CV Accuracies:", cv_scores_cbow)
print("CBOW model - LR Mean 5-fold CV Accuracy:", np.mean(cv_scores_cbow))

# Evaluating Skip-gram model with 5-fold cross-validation
cv_scores_skipgram = cross_val_score(lr_best, X_train_skipgram_vec, y_train, cv=5, scoring='accuracy')
print("Skip-gram model - LR 5-fold CV Accuracies:", cv_scores_skipgram)
print("Skip-gram model - LR Mean 5-fold CV Accuracy:", np.mean(cv_scores_skipgram))

# Test the models on the test set after cross-validation
lr_best.fit(X_train_cbow_vec, y_train)
y_pred_cbow = lr_best.predict(np.array([average_word_vectors(text, cbow_model, 100) for text in X_test]))
print("CBOW model - lr Test Accuracy:", accuracy_score(y_test, y_pred_cbow))

lr_best.fit(X_train_skipgram_vec, y_train)
y_pred_skipgram = lr_best.predict(np.array([average_word_vectors(text, skipgram_model, 100) for text in X_test]))
print("Skip-gram model - lr Test Accuracy:", accuracy_score(y_test, y_pred_skipgram))

# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >-- Experiment 5 - Comparison: Words, NER, YAKE Keyphrases, SpaCy KP ---<
# >-----------------------------------------------------------------------<

# Initialize models and tools
nlp = spacy.load('en_core_web_sm')
kw_extractor = yake.KeywordExtractor()

# Named Entity Extraction
def extract_named_entities(text):
    doc = nlp(text)
    return ' '.join(ent.text for ent in doc.ents)

# YAKE Keyphrase Extraction
def extract_keyphrases_yake(text):
    keywords = kw_extractor.extract_keywords(text)
    return ' '.join(kw[0] for kw in keywords)

# SpaCy Keyphrase Extraction
matcher = Matcher(nlp.vocab)
pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN', 'OP': '?'}]
matcher.add("KEYPHRASE", [pattern])
def extract_keyphrases_spacy(text):
    doc = nlp(text)
    return ' '.join(doc[start:end].text for match_id, start, end in matcher(doc))

# Data Preprocessing and Model Pipelines
preprocessing_techniques = {
    'Only Words': None,  # No preprocessing function
    'Named Entity Extraction': extract_named_entities,
    'YAKE Keyphrase Extraction': extract_keyphrases_yake,
    'SpaCy Keyphrase Extraction': extract_keyphrases_spacy
}

# Splitting original data
X = df['Text']
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Comparison loop
results = []
for name, preprocessor_func in preprocessing_techniques.items():
    vectorizer = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range = (1,3))  # Reinitialize to avoid carryover
    lr = LogisticRegression(C=10, solver = 'liblinear') # Reinitialize to avoid carryover

    if preprocessor_func:
        X_processed = [preprocessor_func(text) for text in X_train]
    else:
        X_processed = X_train

    # Vectorizing the processed text
    X_vectorized = vectorizer.fit_transform(X_processed)
    
    # Performing cross-validation and calculating mean CV score
    cv_scores = cross_val_score(lr, X_vectorized, y_train, cv=5)
    mean_cv_score = np.mean(cv_scores)

    # Collecting results
    results.append({
        'Technique': name,
        'Mean CV Score': mean_cv_score
    })

for result in results:
    print(f"{result['Technique']} - Mean CV Score: {result['Mean CV Score']:.3f}")
# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >-------------- Testing the final models on the test data --------------<
# >-----------------------------------------------------------------------<

# Evaluating final model
X = df['Text']
y = df['Score']

# Final model evaluation without YAKE Keyphrase Extraction

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TfidfVectorizer with the specified parameters
vectorizer_final = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))

# Create a Logistic Regression model with the specified parameters
lr_model_final = LogisticRegression(C=10, solver='liblinear')

# Create a pipeline that first vectorizes the text and then applies the Logistic Regression model
pipeline_final = Pipeline([
    ('tfidf', vectorizer_final),
    ('logreg', lr_model_final)
])

# Fit the pipeline to the training data
pipeline_final.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_final = pipeline_final.predict(X_test)

# Calculate the accuracy of the predictions
accuracy_final = accuracy_score(y_test, y_pred_final)

print(f"Final Model Test Accuracy: {accuracy_final:.3f}")

# Evaluating the model with YAKE Keyphrase Extraction

# Preprocess the test data with YAKE keyphrase extraction
X_test_yake = [extract_keyphrases_yake(text) for text in X_test]

# Fit the pipeline to the training data preprocessed with YAKE keyphrase extraction
X_train_yake = [extract_keyphrases_yake(text) for text in X_train]
pipeline_final.fit(X_train_yake, y_train)

# Predict the labels for the preprocessed test set
y_pred_yake = pipeline_final.predict(X_test_yake)

# Calculate the accuracy of the predictions
accuracy_yake = accuracy_score(y_test, y_pred_yake)

print(f"YAKE Keyphrase Extraction Model Test Accuracy: {accuracy_yake:.3f}")

# >***********************************************************************<