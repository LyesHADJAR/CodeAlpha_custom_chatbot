import nltk 
# nltk.download('punkt') # Uncomment this line if you don't have the punkt package
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

# =================================================================
# The work plan
# 1. Theory and some NLP concepts
# 2. Set the Data
# 3. Setting and training the model using pytorch
# 4. Saving and loading the model
# =================================================================


# 1. Theory and some NLP concepts

# =================================================================
# In this part we want to implement the basic functionalities : 
# 1. Tokenization
# 2. Stemming + Lowercasing
# 3. Remove punctuation
# 4. Remove stopwords
# 5. Remove numbers
# 6. Bag of words
# =================================================================


def tokenize(text):
    return nltk.word_tokenize(text)

def stem(token):
    return stemmer.stem(token.lower())

def bag_of_words(tokenized_text, all_words):
    tokenized_text = [stem(w) for w in tokenized_text]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in tokenized_text:
            bag[i] = 1.0
    return bag    
    
