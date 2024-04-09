import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix



class MultinomialNaiveBayes():
    
    def __init__(self) -> None:
        self.class_probabilities = None #P(C)
        self.feature_probabilities = None #P(f|C)
        self.classes = None
        self.vocab = None

    def preprocess_data(self, data):
        # Create a TF-IDF vectorizer with n-grams and binary=True
        self.vectorizer = TfidfVectorizer(use_idf=True,lowercase=True, ngram_range=(1, 2), binary=True)
        # Fit the vectorizer to the data and transform the data
        X = self.vectorizer.fit_transform(data)
        X_binary = (X != 0).astype(int)
        # Store the vocabulary
        self.vocab = self.vectorizer.get_feature_names_out()
        return X.toarray()
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate class probabilities
        self.class_probabilities = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_probabilities[i] = np.sum(y == c) / n_samples # P(C)

        # Calculate feature probabilities
        self.feature_probabilities = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            class_mask = (y == c) # Sample belong the class c (Boolean)
            # Sums the occurrences of each feature (word) across the documents belonging to class c
            class_word_counts = X[class_mask].sum(axis=0) 
            # Calculates the total count of all features in class c
            total_word_counts = class_word_counts.sum()
            
            self.feature_probabilities[i] = (class_word_counts + 1) / (total_word_counts 
                                                                       + n_features) # Add smoothing
    
    def predict(self, X):
        # Calculate log likelihoods for each class
        log_likelihoods = np.dot(X, np.log(self.feature_probabilities.T))
        log_likelihoods += np.log(self.class_probabilities)
        
        # Make predictions
        predictions = np.argmax(log_likelihoods, axis=1)
        return predictions
    
    pass
## Import data
## Sourse kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/images

data_1 = pd.read_csv('Youtube01-Psy.csv')
data_2 = pd.read_csv('Youtube02-KatyPerry.csv')
data_3 = pd.read_csv('Youtube03-LMFAO.csv')
data_4 = pd.read_csv('Youtube04-Eminem.csv')
data_5 = pd.read_csv('Youtube05-Shakira.csv')
os.system('cls')

data_total = pd.DataFrame()
data_total = pd.concat([data_1,data_2,data_3,data_4,data_5])
# print(data_total.shape)

# Preprocessing data
data_1_ext = data_total[['CONTENT','CLASS']]
# print(data_1_ext.shape)

# Content data
X_data = data_1_ext.iloc[:,0].values # Content
# print(X_data[:10])
y_data = data_1_ext.iloc[:,1] #Class

## Train model

model_2 = MultinomialNaiveBayes()
X_data_encoded = model_2.preprocess_data(X_data)
X_train,X_test,y_train,y_test = train_test_split(X_data_encoded,y_data,
                                                 test_size=0.2,random_state=42)

train = model_2.fit(X_train,y_train)
predict = model_2.predict(X_test)

cm = confusion_matrix(predict,y_test)

sns.heatmap(data = cm,annot = True,fmt='d')

## Testing 1

test_comment1 = "i don't know why this is a ham comment"

# Preprocess the test comment
vocab = model_2.vocab
test_comment1_encoded = np.zeros(len(vocab))
test_words = test_comment1.lower().split()
for word in test_words:
    if word in vocab:
        index = np.where(vocab == word)[0][0]  # Find the index of the word in the vocab
        test_comment1_encoded[index] = 1
      
# Reshape the test comment to match the expected input shape  
test_comment1_encoded = test_comment1_encoded.reshape(1, -1)

print(f'The result of  first test: {model_2.predict(test_comment1_encoded)}')

### Testing 2

comment2 = "do u wanna rent 1000000$? yes? check this https://www.youtube.com/watch?v=fHI8X4OXluQ"

test_comment2_encoded = np.zeros(len(vocab))
test_words = comment2.lower().split()
for word in test_words:
    if word in vocab:
        index = np.where(vocab == word)[0][0]  # Find the index of the word in the vocab
        test_comment2_encoded[index] = 1

test_comment2_encoded = test_comment2_encoded.reshape(1, -1)
print(f'The result of second test: {model_2.predict(test_comment2_encoded)}')



