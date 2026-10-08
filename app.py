import re
import joblib
from nltk.corpus import stopwords

# Load saved model and TF-IDF vectorizer
model = joblib.load("models/logistic_regression_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Load English stopwords
stop_words = set(stopwords.words("english"))


# Function to clean the review
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Function to predict sentiment
def predict_sentiment(review):
    cleaned_review = clean_text(review)

    review_tfidf = tfidf.transform([cleaned_review])

    prediction = model.predict(review_tfidf)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


# Take review from user
review = input("Enter a movie review: ")

result = predict_sentiment(review)

print("\nPredicted Sentiment:", result)