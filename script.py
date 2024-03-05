import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import chunk
import string
import re

# ------------------------------------------
# --- Loading the data and preprocessing ---
# ------------------------------------------

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

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
    # Lowercase conversion
    text = text.lower()
    
    # Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # POS Tagging
    pos_tags = pos_tag(tokens)
    
    # Lemmatization with POS Tags
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    
    # Re-join tokens
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text

def get_wordnet_pos(treebank_tag):
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to NOUN

# Directory path to the dataset containing text documents
directory_path = '../sentiment labelled sentences'  

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
print(df.head(20))
print(df.shape)


# foolin' around
example = df['Text'][40]

print(example)

tokkk = nltk.word_tokenize(example)
print(tokkk)

taggg = nltk.pos_tag(tokkk) 
print(taggg)

entities = nltk.chunk.ne_chunk(taggg)
print (entities)

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

print(sia.polarity_scores(example))
# foolin' around

# ------------------------------------------
# *** Loading the data and preprocessing ***
# ------------------------------------------

# ------------------------------------------
# --------- Training a classifier ----------
# ------------------------------------------

"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(df['Text'], df['Score'], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
y_pred_rf = rf_classifier.predict(X_val_tfidf)

# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred_nb = nb_classifier.predict(X_val_tfidf)

# Logistic Regression Classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_tfidf, y_train)
y_pred_lr = lr_classifier.predict(X_val_tfidf)

# Support Vector Machines Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_val_tfidf)

# Evaluate the performance of the classifiers on the validation set
print("Random Forest Classifier:")
print(classification_report(y_val, y_pred_rf))
print("Accuracy:", accuracy_score(y_val, y_pred_rf))

print("\nNaive Bayes Classifier:")
print(classification_report(y_val, y_pred_nb))
print("Accuracy:", accuracy_score(y_val, y_pred_nb))

print("\nLogistic Regression Classifier:")
print(classification_report(y_val, y_pred_lr))
print("Accuracy:", accuracy_score(y_val, y_pred_lr))

print("\nSupport Vector Machines Classifier:")
print(classification_report(y_val, y_pred_svm))
print("Accuracy:", accuracy_score(y_val, y_pred_svm))


# ------------------------------------------
# --- Hyperoptimization using gridsearch ---
# ------------------------------------------

from sklearn.model_selection import GridSearchCV

# Random Forest grid
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train_tfidf, y_train)

# Naive Bayes grid
nb_params = {
    'alpha': [0.01, 0.1, 1.0, 10.0]
}

nb_grid_search = GridSearchCV(MultinomialNB(), nb_params, cv=5, scoring='accuracy')
nb_grid_search.fit(X_train_tfidf, y_train)

# Logistic regression grid
lr_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

lr_grid_search = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring='accuracy')
lr_grid_search.fit(X_train_tfidf, y_train)

# Support Vector Machine Grid
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm_grid_search = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train_tfidf, y_train)

# Evaluation
print("Best parameters for RF:", rf_grid_search.best_params_)
print("Best score for RF:", rf_grid_search.best_score_)

print("Best parameters for NB:", nb_grid_search.best_params_)
print("Best score for NB:", nb_grid_search.best_score_)

print("Best parameters for LR:", lr_grid_search.best_params_)
print("Best score for LR:", lr_grid_search.best_score_)

print("Best parameters for SVM:", svm_grid_search.best_params_)
print("Best score for SVM:", svm_grid_search.best_score_)
# ------------------------------------------
# *** Hyperoptimization using gridsearch ***
# ------------------------------------------


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrices for all classifiers on the test set
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Plot confusion matrices for all classifiers
plt.figure(figsize=(9, 9))

plt.subplot(2, 2, 1)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random Forest Confusion Matrix')

plt.subplot(2, 2, 2)
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Naive Bayes Confusion Matrix')

plt.subplot(2, 2, 3)
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Logistic Regression Confusion Matrix')

plt.subplot(2, 2, 4)
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Support Vector Machines Confusion Matrix')

plt.tight_layout()
plt.show()

# ------------------------------------------
# ********* Training a classifier **********
# ------------------------------------------
"""