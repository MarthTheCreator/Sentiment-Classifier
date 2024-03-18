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
from nltk import pos_tag

# Import sklkearn libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Import gensim libraries
from gensim.models import Word2Vec

# Import SpaCy and Yake libraries
import spacy
from spacy.matcher import Matcher
import yake

# Import visualization libraries 
import matplotlib.pyplot as plt
import seaborn as sns

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
    return text # returns text without stemming or lemmatization

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

# Model training and predictions for base model
models = {
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting Machine": GradientBoostingClassifier()
}

# Loop through models, perform cross-validation, and print evaluation metrics
for name, model in models.items():
    # Fit the model to the training data
    model.fit(X_train_vectorized, y_train)
    
    # Perform 5-fold cross-validation and calculate the mean CV score
    cv_scores = cross_val_score(model, X_train_vectorized, y_train, cv=5)
    mean_cv_score = np.mean(cv_scores)
    
    # Predict the labels for the test set
    y_pred = model.predict(X_test_vectorized)
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print the results
    print(f"{name} Classifier:")
    print(f"Mean CV Score (5-fold): {mean_cv_score:.3f}")
    print(f"Test Accuracy: {accuracy:.3f}")
    print("\n")

# basic hyperparametertuning grid for base Logistic Regression model
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
# > Experiment 1-3 - Combination: preprocessing, vectorization and n-gram <
# >-----------------------------------------------------------------------<

# Set up a Logistic Regression model with hyperparameters already determined from prior tuning.
lr = LogisticRegression(C=10, solver='liblinear')

# Define the range of n-gram sizes to evaluate. 
# Single words to trigrams are considered to see their impact on model performance.
ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]

# Define preprocessing configurations combining different text preprocessing techniques
# and whether or not to use term frequency-inverse document frequency (TF-IDF) weighting.
preprocessing_configs = [
    ('None', no_preprocessor, 'use_idf=True'),  # No text preprocessing, with TF-IDF weighting
    ('Stemming', stem_preprocessor, 'use_idf=True'),  # Stemming with TF-IDF weighting
    ('Lemmatization', lemma_preprocessor, 'use_idf=True'),  # Lemmatization with TF-IDF weighting
    ('None', no_preprocessor, 'use_idf=False'),  # No text preprocessing, without TF-IDF weighting
    ('Stemming', stem_preprocessor, 'use_idf=False'),  # Stemming without TF-IDF weighting
    ('Lemmatization', lemma_preprocessor, 'use_idf=False')  # Lemmatization without TF-IDF weighting
]

# Results list to store performance metrics for each configuration.
experiment_results = []

# Iterate over each preprocessing and TF-IDF configuration to assess their impact.
for preproc_label, preprocessor_func, tfidf_param in preprocessing_configs:
    # Initialize a TfidfVectorizer based on the current TF-IDF weighting and preprocessing settings.
    if 'True' in tfidf_param:
        vect = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, preprocessor=preprocessor_func) # TF.IDF vectorizer with weights applied
    else:
        vect = TfidfVectorizer(use_idf=False, min_df=0.0, max_df=1.0, preprocessor=preprocessor_func) # TF.IDF vectorizer without weights applied

    # Explore different n-gram ranges for each configuration.
    for ngram_range in ngram_ranges:
        vect.set_params(ngram_range=ngram_range)  # Update the vectorizer's n-gram range.

        # Vectorize the training and test data.
        X_train_vectorized = vect.fit_transform(X_train)
        X_test_vectorized = vect.transform(X_test)

        # Train the Logistic Regression model on the vectorized training data.
        lr.fit(X_train_vectorized, y_train)
        
        # Evaluate model performance using 5-fold cross-validation.
        cv_scores = cross_val_score(lr, X_train_vectorized, y_train, cv=5)
        cv_mean_score = np.mean(cv_scores)  # Average cross-validation score.
        cv_std_dev = np.std(cv_scores)  # Standard deviation of cross-validation scores for stability assessment.
        
        # Assess the model on the test set.
        test_score = lr.score(X_test_vectorized, y_test)

        # Compile and store the results for this configuration.
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

# Convert the experiment results to a DataFrame for better visualization
results_df = pd.DataFrame(experiment_results)

# Sort and rank the DataFrame based on CV mean score
results_df['Rank'] = results_df['Mean CV accuracy'].rank(method='max', ascending=False) # Ranking
results_df = results_df.sort_values(by=['Rank'], ascending=True) # Sorting
print(results_df) # Displaying the dataframe

# Save the results of the experiments as a csv for better visualization
# results_df.to_csv('../experiment1-3_results.csv', index=True)

# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >----- Experiment 4 - BOW representation vs using word embeddings  -----<
# >-----------------------------------------------------------------------<

# Initialize Word2Vec models:
# CBOW model (sg=0): This model predicts the current word based on the context.
cbow_model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, sg=0)

# Skip-gram model (sg=1): This model uses the current word to predict the surrounding context words.
skipgram_model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, sg=1)

# Function to average word vectors for a text.
def average_word_vectors(words, model, num_features):
    # Initialize an empty array for storing word vectors.
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0
    
    # Sum up all vectors for each word in the document if the word is in the model's vocabulary.
    for word in words:
        if word in model.wv.index_to_key:  # Check if word is in the model's vocabulary
            nwords += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    
    # Divide the result by the number of words to get the average.
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

# Convert texts into average word vectors for the training data using both CBOW and Skip-gram.
X_train_cbow_vec = np.array([average_word_vectors(text, cbow_model, 100) for text in X_train])
X_train_skipgram_vec = np.array([average_word_vectors(text, skipgram_model, 100) for text in X_train])

# Use the best Logistic Regression as the classifier
lr_best = LogisticRegression(C=10, solver='liblinear')  # Hyperparameters set from previous tuning

# Evaluate the CBOW model using 5-fold cross-validation.
cv_scores_cbow = cross_val_score(lr_best, X_train_cbow_vec, y_train, cv=5, scoring='accuracy')
mean_cv_score_cbow = np.mean(cv_scores_cbow)
std_cv_score_cbow = np.std(cv_scores_cbow)
print("CBOW model - Logistic Regression Mean 5-fold CV Accuracy:", mean_cv_score_cbow)
print("CBOW model - Logistic Regression Std Dev in 5-fold CV:", std_cv_score_cbow)

# Evaluate the Skip-gram model using the same cross-validation.
cv_scores_skipgram = cross_val_score(lr_best, X_train_skipgram_vec, y_train, cv=5, scoring='accuracy')
mean_cv_score_skipgram = np.mean(cv_scores_skipgram)
std_cv_score_skipgram = np.std(cv_scores_skipgram)
print("Skip-gram model - Logistic Regression Mean 5-fold CV Accuracy:", mean_cv_score_skipgram)
print("Skip-gram model - Logistic Regression Std Dev in 5-fold CV:", std_cv_score_skipgram)
# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >-- Experiment 5 - Only words, Keyphrase extraction and named entity ---<
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
    'Only Words': None,  # No preprocessing function (using only words)
    'Named Entity Extraction': extract_named_entities, # Including SpaCy named entity extraction 
    'YAKE Keyphrase Extraction': extract_keyphrases_yake, # Including YAKE keyphrase extraction
    'SpaCy Keyphrase Extraction': extract_keyphrases_spacy # Including SpaCy keyphrase extraction
}

# Comparison loop
results = []
for name, preprocessor_func in preprocessing_techniques.items():
    vectorizer = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range = (1,3))  # Reinitialize to avoid carryover
    lr = LogisticRegression(C=10, solver = 'liblinear') # Reinitialize to avoid carryover
    # Applying Keyphrase extraction, named entity extraction or using only words. 
    if preprocessor_func:
        X_processed = [preprocessor_func(text) for text in X_train]
    else:
        X_processed = X_train

    # Vectorizing the processed text
    X_vectorized = vectorizer.fit_transform(X_processed)
    
    # Performing cross-validation and calculating mean CV score and standard deviation
    cv_scores = cross_val_score(lr, X_vectorized, y_train, cv=5)
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)

    # Collecting and printing results
    results.append({
        'Technique': name,
        'Mean CV Score': mean_cv_score,
        'CV Score Std Dev': std_cv_score
    })

for result in results:
    print(f"{result['Technique']} - Mean CV Score: {result['Mean CV Score']:.3f}, CV Score Std Dev: {result['CV Score Std Dev']:.3f}")
# >***********************************************************************<


# >-----------------------------------------------------------------------<
# >-------------- Testing the final models on the test data --------------<
# >-----------------------------------------------------------------------<
# Function to perform cross-validation and return mean CV score and standard deviation
def evaluate_model_cv(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv)
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    return mean_cv_score, std_cv_score

# Function for confusion matrix configs
def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()    

# Final model evaluation without YAKE Keyphrase Extraction
vectorizer_final = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 3))
lr_model_final = LogisticRegression(C=10, solver='liblinear')
pipeline_final = Pipeline([('tfidf', vectorizer_final), ('logreg', lr_model_final)])

# Perform cross-validation and print the results
mean_cv_score_final, std_cv_score_final = evaluate_model_cv(pipeline_final, X_train, y_train)
print(f"Final Model - Mean CV Score: {mean_cv_score_final:.3f}, CV Score Std Dev: {std_cv_score_final:.3f}")

# Fit the pipeline to the training data and make predictions on test set
pipeline_final.fit(X_train, y_train)
y_pred_final = pipeline_final.predict(X_test)

# Print classification report and accuracy
accuracy_final = accuracy_score(y_test, y_pred_final)
print("Final Model Classification Report:\n", classification_report(y_test, y_pred_final))
print(f"Final Model Test Accuracy: {accuracy_final:.3f}")

# Plot the confusion matrix for the final model
plot_confusion_matrix(y_test, y_pred_final, "Final Model without YAKE")

# Evaluating the model with YAKE Keyphrase Extraction
X_train_yake = [extract_keyphrases_yake(text) for text in X_train]
X_test_yake = [extract_keyphrases_yake(text) for text in X_test]

# Perform cross-validation and print the results for YAKE preprocessing
mean_cv_score_yake, std_cv_score_yake = evaluate_model_cv(pipeline_final, X_train_yake, y_train)
print(f"YAKE Keyphrase Extraction Model - Mean CV Score: {mean_cv_score_yake:.3f}, CV Score Std Dev: {std_cv_score_yake:.3f}")

# Fit the pipeline to the YAKE preprocessed training data and make predictions on test set
pipeline_final.fit(X_train_yake, y_train)
y_pred_yake = pipeline_final.predict(X_test_yake)

# Print classification report and accuracy
accuracy_yake = accuracy_score(y_test, y_pred_yake)
print("YAKE Keyphrase Extraction Model Classification Report:\n", classification_report(y_test, y_pred_yake))
print(f"YAKE Keyphrase Extraction Model Test Accuracy: {accuracy_yake:.3f}")

# Plot the confusion matrix for the YAKE keyphrase extraction model
plot_confusion_matrix(y_test, y_pred_yake, "YAKE Keyphrase Extraction Model")

# >***********************************************************************<