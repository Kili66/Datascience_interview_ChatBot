import nltk 
import numpy as np
# download pretrained nltk tokenizer
nltk.download('punkt')
# import an stemmer
from nltk.stem.porter import PorterStemmer 
stemmer= PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def steming(word):
    return stemmer.stem(word.lower())

def bag_of_word(sentence_tokenize, all_words):
    """ 
    sentence=["hello", "how", "are", "you"]
    words=["hi", "hello", "I", "are", "you","bye", "thank", "cool"]
    bag=[0, 1, 0, 1, 1, 0, 0, 0]
    """
    bag= np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_tokenize:
            bag[idx]= 1.0
        else:
            bag[idx]=0.0
    return bag    

# sentence=["hello", "how", "are", "you"]
# words=["hi", "hello", "I", "are", "you","bye", "thank", "cool"]
# bow= bag_of_word(sentence, words)
# print(bow)