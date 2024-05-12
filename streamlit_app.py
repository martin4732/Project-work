
import streamlit as st
from PyPDF2 import PdfReader
from tokenization import tokenize_text
from cleaning import clean_tokens
from normalization import normalize_text
from stopword_removal import remove_stopwords
from stemming import stem_tokens
from pos_tagging import pos_tagging  # Import the POS tagging function
from naive_bayes import train_naive_bayes, predict_sentiment

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    # Tokenization
    tokens = tokenize_text(text)
    
    # Cleaning
    cleaned_tokens = clean_tokens(tokens)
    cleaned_text = " ".join(cleaned_tokens)
    
    # Normalization
    normalized_text = normalize_text(cleaned_text)
    
    # Stopword removal
    text_without_stopwords = remove_stopwords(normalized_text)
    
    # Stemming
    stemmed_text = stem_tokens(text_without_stopwords)
    
    # Convert list of stemmed tokens to string
    preprocessed_text = " ".join(stemmed_text)
    
    return preprocessed_text

def main():
    st.title('AI Sentimental Analysis Tool')
    st.markdown('---')
    st.sidebar.title('Upload Documents Section')
    uploaded_files = st.sidebar.file_uploader("Upload Speeches", type=['txt', 'pdf'], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            if file.type == "text/plain":  # For text files
                speech = file.read().decode("utf-8")
            elif file.type == "application/pdf":  # For PDF files
                speech = extract_text_from_pdf(file)
            
            st.write(f"## Analyzing {file.name}")
            st.write("### Preprocessed Speech Text:")
            preprocessed_speech = preprocess_text(speech)
            
            # POS tagging
            pos_tags = pos_tagging(preprocessed_speech)
            st.write("### POS Tagged Text:")
            st.write(pos_tags)

if __name__ == "__main__":
    main()
