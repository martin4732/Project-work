from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_naive_bayes(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate the model
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, report

def predict_sentiment(model, text):
    # Vectorize the input text
    vectorizer = CountVectorizer()
    text_vectorized = vectorizer.transform([text])
    
    # Predict the sentiment
    sentiment = model.predict(text_vectorized)[0]
    
    return sentiment
