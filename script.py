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
model_vectorizer = TfidfVectorizer()
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
    y_pred = model.predict(X_test_vectorized)  # Changed from X_val_vectorized to X_test_vectorized
    print(f"{name} Classifier:")
    #print(classification_report(y_test, y_pred))  # Changed from y_val to y_test
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Changed from y_val to y_test
    print("\n")


'''
# Hyperparameter tuning for Logistic Regression

# Hyperparameter grid
param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}

# Setting up the Logistic Regression model
log_reg_model = LogisticRegression()

# Setting up GridSearchCV
grid_search_lr = GridSearchCV(log_reg_model, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train_vectorized, y_train)  

# Storing the best Logistic Regression model, parameters, and score
best_lr_model = grid_search_lr.best_estimator_
best_lr_params = grid_search_lr.best_params_
best_lr_score = grid_search_lr.best_score_

# Output the best parameters and best cross-validation score
print("Best parameters for Logistic Regression:", best_lr_params)
print("Best cross-validation score for Logistic Regression:", best_lr_score)

# Fit the best Logistic Regression model to the training data and predict on the test set
best_lr_model.fit(X_train_vectorized, y_train)
y_pred_test_lr = best_lr_model.predict(X_test_vectorized)  # Use the best model to predict the test set
print("Test Accuracy with Logistic Regression:", accuracy_score(y_test, y_pred_test_lr))

# Best hyperparameter
#lg = LogisticRegression(C=1, solver='liblinear')
lg = SVC()
# >***********************************************************************<
'''

# Hyperparameter grids
param_grids = {
    "Naive Bayes": {'alpha': [0.01, 0.1, 1, 10]},
    "Logistic Regression": {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
    "Support Vector Machine": {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
}

# Model training and predictions
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC()
}

best_models = {}
best_params = {}
best_scores = {}

# Looping through the models with GridSearch
for name, model in models.items():
    print(f"\nStarting Grid Search for {name}")
    print(f"Parameters being tested: {param_grids[name]}")
    
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_vectorized, y_train)  
    
    best_models[name] = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_
    best_scores[name] = grid_search.best_score_
    
    print(f"Best parameters for {name}: {best_params[name]}")
    print(f"Best cross-validation score for {name}: {best_scores[name]:.3f}\n")

# Loop through best models, fit, predict, and print evaluation metrics
for name, model in best_models.items():
    model.fit(X_train_vectorized, y_train)
    y_pred_test = model.predict(X_test_vectorized)  # Changed from X_val_vectorized to X_test_vectorized
    print(f"{name} Classifier with Best Parameters:")
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))  # Changed from y_val to y_test
    print("\n")

lg = SVC(C = 10, gamma = 'scale', kernel = 'rbf')

# >-----------------------------------------------------------------------<
# >--------------------------- Experiment 1-3 ----------------------------<
# 1. Stemming, lemmatization or neither
# 2. With tf.idf weigths or without tf.idf weights
# 3. Using complete words or n_grams 
# >-----------------------------------------------------------------------<

# Define the parameter grid for n-gram ranges and vectorization strategies
ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]
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
        #print(f'Using {preproc_label} preprocessing with TF-IDF {tfidf_param} and n-gram range: {ngram_range}')
        
        # Update vectorizer n-gram range
        vect.set_params(ngram_range=ngram_range)
        
        # Fit and transform the training data with the vectorizer
        X_train_vectorized = vect.fit_transform(X_train)
        X_test_vectorized = vect.transform(X_test)  # Changed from X_val to X_test
        
        # Train and evaluate the Logistic Regression model
        lg.fit(X_train_vectorized, y_train)
        cv_scores = cross_val_score(lg, X_train_vectorized, y_train, cv=5)
        cv_mean_score = np.mean(cv_scores)
        test_score = lg.score(X_test_vectorized, y_test)  # Changed from val_score to test_score

        # Store the performance metrics
        experiment_results.append({
            'Preprocessing': preproc_label,
            'TF-IDF': tfidf_param,
            'N-gram range': ngram_range,
            'CV accuracy': cv_scores,
            'Mean CV accuracy': cv_mean_score,
            'Test accuracy': test_score  
        })

        # Print the performance metrics
        #print('CV accuracy (5-fold):', cv_scores)
        #print('Mean CV accuracy (5-fold):', cv_mean_score)
        #print('Test accuracy:', test_score)  # Changed from Validation to Test
        

# Print summary of results for all configurations
for result in experiment_results:
    print(f"{result['Preprocessing']} preprocessing with TF-IDF {result['TF-IDF']} and n-gram range {result['N-gram range']} - "
          f"Mean CV accuracy: {result['Mean CV accuracy']}, Test accuracy: {result['Test accuracy']}")
# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >----- Experiment 4 - BOW representation vs using word embeddings  -----<
# >-----------------------------------------------------------------------<

# Convert processed_texts from a list of strings to a list of word lists
tokenized_texts = [text.split() for text in processed_texts]

# For this example, we're training a new model
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Save the model for later use
word2vec_model.save("word2vec_text.model")

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

# We need to split the tokenized_texts according to the original train-test split
tokenized_texts_train = [text.split() for text in X_train]
tokenized_texts_test = [text.split() for text in X_test]  # Change from val to test

# Generate the average word vectors for the training and test sets
X_train_word2vec = np.array([average_word_vectors(text, word2vec_model, set(word2vec_model.wv.index_to_key), 100) for text in tokenized_texts_train])
X_test_word2vec = np.array([average_word_vectors(text, word2vec_model, set(word2vec_model.wv.index_to_key), 100) for text in tokenized_texts_test])  # Change from val to test

# Now, fit the model with X_train_word2vec and y_train
lg.fit(X_train_word2vec, y_train)

# Predict and evaluate the model with X_test_word2vec and y_test  # Change from val to test
y_pred_word2vec = lg.predict(X_test_word2vec)  # Change from val to test
print("Word2Vec Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_word2vec))  # Change from val to test
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
    vectorizer = TfidfVectorizer()  # Reinitialize to avoid carryover
    #lg = LogisticRegression()
    lg = SVC()

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', lg)
    ])
    
    # Apply preprocessing if the technique requires it
    if preprocessor_func:
        X_train_processed = [preprocessor_func(text) for text in X_train]
        X_test_processed = [preprocessor_func(text) for text in X_test]
    else:
        X_train_processed, X_test_processed = X_train, X_test

    # Fit and evaluate
    pipeline.fit(X_train_processed, y_train)
    y_pred = pipeline.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)

    # Collecting results
    results.append({
        'Technique': name,
        'Accuracy': accuracy
    })

for result in results:
    print(f"{result['Technique']} - Accuracy: {result['Accuracy']}")
# >***********************************************************************<