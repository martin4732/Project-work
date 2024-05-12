import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only needs to be done once)
nltk.download('stopwords')

# Function for stopword removal
def remove_stopwords(normalized_tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in normalized_tokens if token not in stop_words]
    return filtered_tokens
