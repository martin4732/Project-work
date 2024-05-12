import re

# Function for cleaning text
def clean_tokens(tokens):
    cleaned_tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
    cleaned_tokens = [token for token in cleaned_tokens if token.strip()]
    return cleaned_tokens
