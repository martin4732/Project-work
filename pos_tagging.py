import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def pos_tagging(preprocessed_text):
    # Tokenization (if not already tokenized)
    tokens = word_tokenize(preprocessed_text)
    
    # POS tagging
    pos_tags = pos_tag(tokens)
    
    return pos_tags
