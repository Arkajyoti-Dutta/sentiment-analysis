print("========== SENTIMENT ANALYSIS SYSTEM ==========\n")

import pandas as pd

# Load Dataset
data = pd.read_csv("IMDB Dataset.csv")

# Convert labels
data["sentiment"] = data["sentiment"].replace({
    "positive": 1,
    "negative": 0
})

data["sentiment"] = data["sentiment"].astype(int)

# Features and Labels
x = data["review"]
y = data["sentiment"]

# Train-Test Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)

print("Dataset Loaded Successfully")
print("Training Data:", len(x_train))
print("Testing Data:", len(x_test))

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

print("Text Vectorization Completed")

# Train Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(xv_train, y_train)

print("Model Training Completed")

# Model Accuracy
from sklearn.metrics import accuracy_score

predictions = model.predict(xv_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# User Input
print("\n====================================")
text = input("Enter Text: ")

text_vector = vectorizer.transform([text])

prediction = model.predict(text_vector)

# Confidence Scores
probabilities = model.predict_proba(text_vector)

positive_prob = probabilities[0][1]
negative_prob = probabilities[0][0]

print("\n========== RESULT ==========")

print("Positive Probability:", positive_prob)
print("Negative Probability:", negative_prob)

difference = abs(positive_prob - negative_prob)

print("Difference:", difference)

if difference < 0.10:
    print("😐 NEUTRAL SENTIMENT")
    print("The entered text appears neutral.")
    print(f"Confidence: {max(positive_prob, negative_prob) * 100:.2f}%")

elif prediction[0] == 1:
    print("😊 POSITIVE SENTIMENT")
    print("The entered text expresses a positive opinion.")
    print(f"Confidence: {positive_prob * 100:.2f}%")

else:
    print("😞 NEGATIVE SENTIMENT")
    print("The entered text expresses a negative opinion.")
    print(f"Confidence: {negative_prob * 100:.2f}%")

print("\nThank you for using the Sentiment Analysis System.")

