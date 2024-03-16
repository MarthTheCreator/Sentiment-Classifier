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

# >------------------------------------------<
# >-- Functions and controls of the script --<
# >------------------------------------------<

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
    else:  # Default case
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


# >******************************************<
# >** Functions and controls of the script **<
# >******************************************<

# >------------------------------------------<
# >--- Loading the data and preprocessing ---<
# >------------------------------------------<

# Load all three datasets
directory_path = '../sentiment labelled sentences'
all_data = load_data(directory_path)

# Applying the preprocess_text function to each text in all_data
processed_texts = [' '.join(preprocess_text(text)) for text, _ in all_data]

# Prepare the dataset
df = pd.DataFrame({'Text': processed_texts, 'Score': [score for _, score in all_data]})

# >******************************************<
# >*** Loading the data and preprocessing ***<
# >******************************************<

# >------------------------------------------<
# >--------- Training a classifier ----------<
# >------------------------------------------<

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Score'], test_size=0.2, random_state=42)

# Vectorizing the data for basic model building. 
model_vectorizer = CountVectorizer(binary=False)
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

# >******************************************<
# >********* Training a classifier **********<
# >******************************************<


# >------------------------------------------<
# >--------- Hyperparameter tuning ----------<
# >------------------------------------------<

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



# Best hyperparameter
lg = LogisticRegression(C=1, solver='liblinear')

# >******************************************<
# >********* Hyperparameter tuning **********<
# >******************************************<
'''
lg = LogisticRegression()

'''
'''
# >-----------------------------------------------------------------------<
# Experiment 1 - Using stemming, lemmatization or neither in preprocessing.
# >-----------------------------------------------------------------------<

preprocessing_results = []
preprocessing_configs = [
    ('None', no_preprocessor),
    ('Stemming', stem_preprocessor),
    ('Lemmatization', lemma_preprocessor)
]

for label, preprocessor_func in preprocessing_configs:
    print(f'Using preprocessing: {label}')

    # Create CountVectorizer with specific preprocessor
    vectorizer = CountVectorizer(preprocessor=preprocessor_func)
    train_features = vectorizer.fit_transform(X_train)
    val_features = vectorizer.transform(X_val)
    
    # Train and evaluate the Logistic Regression model
    lg.fit(train_features, y_train)
    preprocessing_cv_scores = cross_val_score(lg, train_features, y_train, cv=5)
    preprocessing_cv_mean_score = np.mean(preprocessing_cv_scores)
    preprocessing_val_score = lg.score(val_features, y_val)

    # Store and print the results
    preprocessing_results.append({
        'preprocessing': label,
        'cv_accuracy': preprocessing_cv_scores,
        'mean_cv_accuracy': preprocessing_cv_mean_score,
        'validation_accuracy': preprocessing_val_score
    })
    print('CV accuracy (5-fold):', preprocessing_cv_scores)
    print('Mean CV accuracy (5-fold):', preprocessing_cv_mean_score)
    print('Validation accuracy:', preprocessing_val_score)
    print()

# Print summary
for result in preprocessing_results:
    print(f"Preprocessing {result['preprocessing']} - Mean CV accuracy: {result['mean_cv_accuracy']}, Validation accuracy: {result['validation_accuracy']}")


# >***********************************************************************<
# >******* Experiment 1 - Using stemming, lemmatization or neither *******<
# >***********************************************************************<
'''
'''
# >-----------------------------------------------------------------------<
# >----- Experiment 2 - Using TF.IDF with weights or without weights -----<
# >-----------------------------------------------------------------------<

# Bag of words model (TF - term frequency)
print('BOW model')
# Build BOW features on train articles
cv = CountVectorizer(binary=True, min_df=0.0, max_df=1.0)
cv_train_features = cv.fit_transform(X_train)  # Fit and transform the training data

# Transform validation articles into features
cv_val_features = cv.transform(X_val)  # Only transform the validation data
print('BOW model: train features shape', cv_train_features.shape, 'validation features shape:', cv_val_features.shape)

# Logistic Regression with BOW
lg.fit(cv_train_features, y_train)
lg_bow_cv_scores = cross_val_score(lg, cv_train_features, y_train, cv=5)
lg_bow_cv_mean_score = np.mean(lg_bow_cv_scores)
print('CV accuracy (5-fold):', lg_bow_cv_scores)
print('Mean CV accuracy (5-fold):', lg_bow_cv_mean_score)
lg_bow_val_score = lg.score(cv_val_features, y_val)
print('Accuracy', lg_bow_val_score)

# Bag of words model (TF.IDF - term frequency inverse document frequency)
print('TF.IDF model')
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
tv_train_features = tv.fit_transform(X_train)

# Transform validation articles into features
tv_val_features = tv.transform(X_val)  # Only transform the validation data
print('TF.IDF model: train features shape', tv_train_features.shape, 'validation features shape:', tv_val_features.shape)

# Logistic Regression with TF.IDF
lg.fit(tv_train_features, y_train)
lg_tfidf_tv_scores = cross_val_score(lg, tv_train_features, y_train, cv=5)
lg_tfidf_tv_mean_score = np.mean(lg_tfidf_tv_scores)
print('TV accuracy (5-fold):', lg_tfidf_tv_scores)
print('Mean CV accuracy (5-fold):', lg_tfidf_tv_mean_score)
lg_tfidf_val_score = lg.score(tv_val_features, y_val)
print('Accuracy', lg_tfidf_val_score)

# >***********************************************************************<
# >***** Experiment 2 - Using TF.IDF with weights or without weights *****<
# >***********************************************************************<
'''
'''
# >-----------------------------------------------------------------------<
# Experiment 2 - Integrating preprocessing and TF.IDF with or without weights.
# >-----------------------------------------------------------------------<

tfidf_results = []
preprocessing_configs = [
    ('None_TFIDF_True', no_preprocessor, TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)),
    ('Stemming_TFIDF_True', stem_preprocessor, TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)),
    ('Lemmatization_TFIDF_True', lemma_preprocessor, TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)),
    ('None_TFIDF_False', no_preprocessor, TfidfVectorizer(use_idf=False, min_df=0.0, max_df=1.0)),
    ('Stemming_TFIDF_False', stem_preprocessor, TfidfVectorizer(use_idf=False, min_df=0.0, max_df=1.0)),
    ('Lemmatization_TFIDF_False', lemma_preprocessor, TfidfVectorizer(use_idf=False, min_df=0.0, max_df=1.0))
]

for label, preprocessor_func, vectorizer in preprocessing_configs:
    print(f'Using {label} preprocessing with {vectorizer.__class__.__name__}')

    if preprocessor_func:
        vectorizer.set_params(preprocessor=preprocessor_func)
    train_features = vectorizer.fit_transform(X_train)
    test_features = vectorizer.transform(X_test)  # Changed from X_val to X_test
    
    lg.fit(train_features, y_train)
    tfidf_cv_scores = cross_val_score(lg, train_features, y_train, cv=5)
    tfidf_cv_mean_score = np.mean(tfidf_cv_scores)
    tfidf_test_score = lg.score(test_features, y_test)  # Changed from tfidf_val_score to tfidf_test_score

    tfidf_results.append({
        'Preprocessing': label,
        'Vectorizer': vectorizer.__class__.__name__,
        'CV Accuracy': tfidf_cv_scores,
        'Mean CV Accuracy': tfidf_cv_mean_score,
        'Test Accuracy': tfidf_test_score  # Changed from 'Validation Accuracy' to 'Test Accuracy'
    })
    print('CV accuracy (5-fold):', tfidf_cv_scores)
    print('Mean CV accuracy (5-fold):', tfidf_cv_mean_score)
    print('Test accuracy:', tfidf_test_score)  # Changed from 'Validation accuracy' to 'Test accuracy'

# Print summary
for result in tfidf_results:
    print(f"Preprocessing {result['Preprocessing']} with {result['Vectorizer']} - Mean CV accuracy: {result['Mean CV Accuracy']}, Test accuracy: {result['Test Accuracy']}")
'''



# >-----------------------------------------------------------------------<
# >-------- Experiment 1-3 - using complete words vs using n_grams ---------<
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
            'Test accuracy': test_score  # Changed from Validation to Test
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
# >******** Experiment 1-3 - using complete words vs using n_grams *********<
# >***********************************************************************<


'''

# >-----------------------------------------------------------------------<
# >-------- Experiment 3 - using complete words vs using n_grams ---------<
# >-----------------------------------------------------------------------<

# Define the parameter grid for n-gram ranges and vectorization strategies
ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]
vectorizers = {
    'CountVectorizer': cv,
    'TfidfVectorizer': tv_
}

# Store results
experiment_results = []

# Loop through each vectorizer and n-gram range
for vect_name, vect in vectorizers.items():
    for ngram_range in ngram_ranges:
        print(f'Using {vect_name} with n-gram range: {ngram_range}')
        
        # Update vectorizer n-gram range
        vect.set_params(ngram_range=ngram_range)
        
        # Create a pipeline with the current vectorizer and Logistic Regression
        pipeline = Pipeline([
            ('vect', vect),
            ('lg', lg)
        ])
        
        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        # Evaluate on the validation set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        cv_mean_score = np.mean(cv_scores)
        val_score = pipeline.score(X_val, y_val)

        # Store the performance metrics
        experiment_results.append({
            'Vectorizer': vect_name,
            'N-gram range': ngram_range,
            'CV accuracy': cv_scores,
            'Mean CV accuracy': cv_mean_score,
            'Validation accuracy': val_score
        })

        # Print the performance metrics
        print('CV accuracy (5-fold):', cv_scores)
        print('Mean CV accuracy (5-fold):', cv_mean_score)
        print('Validation accuracy:', val_score)
        print()

# Print summary of results for all configurations
for result in experiment_results:
    print(f"{result['Vectorizer']} with n-gram range {result['N-gram range']} - "
          f"Mean CV accuracy: {result['Mean CV accuracy']}, "
          f"Validation accuracy: {result['Validation accuracy']}")

# >***********************************************************************<
# >******** Experiment 3 - using complete words vs using n_grams *********<
# >***********************************************************************<
'''

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

'''

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

# We need to split the tokenized_texts according to the original train-validation split
tokenized_texts_train = [text.split() for text in X_train]
tokenized_texts_val = [text.split() for text in X_val]

# Generate the average word vectors for the training and validation sets
X_train_word2vec = np.array([average_word_vectors(text, word2vec_model, set(word2vec_model.wv.index_to_key), 100) for text in tokenized_texts_train])
X_val_word2vec = np.array([average_word_vectors(text, word2vec_model, set(word2vec_model.wv.index_to_key), 100) for text in tokenized_texts_val])

# Now, fit the model with X_train_word2vec and y_train
lg.fit(X_train_word2vec, y_train)

# Predict and evaluate the model with X_val_word2vec and y_val
y_pred_word2vec = lg.predict(X_val_word2vec)
print("Word2Vec Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_word2vec))

# >***********************************************************************<
# >***** Experiment 4 - BOW representation vs using word embeddings ******<
# >***********************************************************************<

'''
'''
# >-----------------------------------------------------------------------<
# >---------- Experiment 5 - Only words vs keyphrase extraction ----------<
# >-----------------------------------------------------------------------<
import spacy
from spacy.matcher import Matcher
import yake


nlp = spacy.load('en_core_web_sm')  # Loading spaCy's English language model
kw_extractor = yake.KeywordExtractor()  # Initialize YAKE

def extract_keyphrases_yake(text):
    keywords = kw_extractor.extract_keywords(text)
    keyphrases = [kw[0] for kw in keywords]
    return ' '.join(keyphrases)

matcher = Matcher(nlp.vocab)

# Define pattern for keyphrase (example: adjective followed by one or two nouns)
pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN', 'OP': '?'}]
matcher.add("KEYPHRASE", [pattern])

def extract_keyphrases_spacy(text):
    doc = nlp(text)
    matches = matcher(doc)
    keyphrases = set()
    for match_id, start, end in matches:
        span = doc[start:end]  # The matched span
        keyphrases.add(span.text)
    return ' '.join(keyphrases)


df['Keyphrases'] = df['Text'].apply(extract_keyphrases_yake)  # or use extract_keyphrases_spacy


# Splitting data for keyphrase experiment
X_train, X_test, y_train, y_test = train_test_split(df['Keyphrases'], df['Score'], test_size=0.4, random_state=42)

# Vectorizing keyphrases
X_train_vect = model_vectorizer.fit_transform(X_train)
X_test_vect = model_vectorizer.transform(X_test)

# Training and evaluating Logistic Regression model
lg.fit(X_train_vect, y_train)
y_pred = lg.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))


# >***********************************************************************<
# >********** Experiment 5 - Only words vs keyphrase extraction **********<
# >***********************************************************************<
'''

# >-----------------------------------------------------------------------<
# >-- Experiment 5 - Comparison: Words, NER, YAKE Keyphrases, SpaCy KP --<
# >-----------------------------------------------------------------------<
import spacy
from spacy.matcher import Matcher
import yake
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
    vectorizer = CountVectorizer()  # Reinitialize to avoid carryover
    lg = LogisticRegression()

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

# >-----------------------------------------------------------------------<
# >------------------ Displaying results of Experiment 5 -----------------<
# >-----------------------------------------------------------------------<
for result in results:
    print(f"{result['Technique']} - Accuracy: {result['Accuracy']}")
