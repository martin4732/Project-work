import string

# Function for text normalization
def normalize_text(tokens):
    normalized_tokens = [token.lower().strip(string.punctuation) for token in tokens]
    return normalized_tokens
