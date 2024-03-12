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

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(df['Text'], df['Score'], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Vectorizing the data for basic model building. 
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

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
    y_pred = model.predict(X_val_vectorized)
    print(f"{name} Classifier:")
    #print(classification_report(y_val, y_pred))
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\n")

# >******************************************<
# >********* Training a classifier **********<
# >******************************************<


# >------------------------------------------<
# >--------- Hyperparameter tuning ----------<
# >------------------------------------------<

# Hyperparameter grids
param_grids = {
    "Naive Bayes": {
        'alpha': [0.01, 0.1, 1, 10]
    },
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
}

# Model training and predictions
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
}

best_models = {}
best_params = {}
best_scores = {}

# Looping through the models with GridSearch
for name, model in models.items():
    # Print the name of the model and the parameters being tested
    print(f"\nStarting Grid Search for {name}")
    print(f"Parameters being tested: {param_grids[name]}")
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_vectorized, y_train)  
    
    # Store and print the best found settings
    best_models[name] = grid_search.best_estimator_  # Best model after GridSearch
    best_params[name] = grid_search.best_params_     # Best parameters found by GridSearch
    best_scores[name] = grid_search.best_score_      # Best cross-validation score
    print(f"Best parameters for {name}: {best_params[name]}")
    print(f"Best cross-validation score for {name}: {best_scores[name]:.3f}")
    print("\n")

# Loop through best models, fit, predict, and print evaluation metrics
for name, model in best_models.items():  # Using best_models here
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_val_vectorized)
    print(f"{name} Classifier with Best Parameters:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\n")

# Best hyperparameter
mnb = MultinomialNB(alpha=1)

# >******************************************<
# >********* Hyperparameter tuning **********<
# >******************************************<



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
    
    # Train and evaluate the Naive Bayes model
    mnb.fit(train_features, y_train)
    preprocessing_cv_scores = cross_val_score(mnb, train_features, y_train, cv=5)
    preprocessing_cv_mean_score = np.mean(preprocessing_cv_scores)
    preprocessing_val_score = mnb.score(val_features, y_val)

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

# TF Multinomial Naïve Bayes
mnb.fit(cv_train_features,y_train)
mnb_bow_cv_scores = cross_val_score(mnb, cv_train_features, y_train, cv=5)
mnb_bow_cv_mean_score = np.mean(mnb_bow_cv_scores)
print('CV accuracy (5-fold):', mnb_bow_cv_scores)
print('Mean CV accuracy (5-fold):', mnb_bow_cv_mean_score)
mnb_bow_val_score = mnb.score(cv_val_features,y_val)
print('Accuracy', mnb_bow_val_score)

# Bag of words model (TF.IDF - term frequency inverse document frequency)
print('TF.IDF model')
tv = TfidfVectorizer(use_idf = True, min_df = 0.0, max_df = 1.0)
tv_train_features = tv.fit_transform(X_train)

# Transform validation articles into features
tv_val_features = tv.transform(X_val)  # Only transform the validation data
print('TF.IDF model: train features shape', tv_train_features.shape, 'validation features shape:', tv_val_features.shape)

# TF.IDF Multinomial Naïve Bayes
mnb.fit(tv_train_features,y_train)
mnb_tfidf_tv_scores = cross_val_score(mnb, tv_train_features, y_train, cv=5)
mnb_tfidf_tv_mean_score = np.mean(mnb_tfidf_tv_scores)
print('tV accuracy (5-fold):', mnb_tfidf_tv_scores)
print('Mean CV accuracy (5-fold):', mnb_tfidf_tv_mean_score)
mnb_tfidf_val_score = mnb.score(tv_val_features,y_val)
print('Accuracy', mnb_tfidf_val_score)

# >***********************************************************************<
# >***** Experiment 2 - Using TF.IDF with weights or without weights *****<
# >***********************************************************************<

# >-----------------------------------------------------------------------<
# >-------- Experiment 3 - using complete words vs using n_grams ---------<
# >-----------------------------------------------------------------------<

# Define the parameter grid for n-gram ranges and vectorization strategies
ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]
vectorizers = {
    'CountVectorizer': CountVectorizer(binary=True, min_df=0.0, max_df=1.0),
    'TfidfVectorizer': TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
}

# Store results
experiment_results = []

# Loop through each vectorizer and n-gram range
for vect_name, vect in vectorizers.items():
    for ngram_range in ngram_ranges:
        print(f'Using {vect_name} with n-gram range: {ngram_range}')
        
        # Update vectorizer n-gram range
        vect.set_params(ngram_range=ngram_range)
        
        # Create a pipeline with the current vectorizer and MultinomialNB
        pipeline = Pipeline([
            ('vect', vect),
            ('mnb', mnb)
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