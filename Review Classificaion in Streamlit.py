import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset with updated cache function
@st.cache_data
def load_data():
    df = pd.read_csv('fake reviews dataset.csv')
    df['label'] = df['label'].map({'CG': 1, 'OR': 0})
    return df

df = load_data()

# The rest of your code remains the same...


# Define the feature (X) and target (y) variables
X = df['text_']  # The review text
y = df['label']  # The labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Convert the text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

# Function to predict whether a review is fake or genuine using Logistic Regression
def predict_review(review_text):
    review_tfidf = vectorizer.transform([review_text])
    prediction = model.predict(review_tfidf)
    label = 'CG' if prediction[0] == 1 else 'OR'
    return label

# Streamlit UI
st.title("Review Classification")
st.write("Enter a review below to classify whether it's fake (CG) or genuine (OR):")

user_review = st.text_area("Review Text", "")

if st.button("Classify"):
    if user_review:
        result = predict_review(user_review)
        st.write(f'The review is classified as: {result}')
    else:
        st.write("Please enter a review to classify.")
