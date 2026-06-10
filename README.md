# Sentiment Analysis using Machine Learning

## Project Overview

This project is a Machine Learning-based Sentiment Analysis System developed using Python and Scikit-Learn. The model analyzes user-entered text and predicts whether the sentiment expressed is Positive or Negative.

The project uses the IMDb Movie Reviews Dataset and applies Natural Language Processing (NLP) techniques to convert text into numerical features before training a Logistic Regression classifier.

---

## Features

* Sentiment Classification (Positive / Negative)
* TF-IDF Text Vectorization
* Logistic Regression Model
* User Input Prediction
* Confidence Score Display
* Model Accuracy Evaluation

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* TF-IDF Vectorizer
* Logistic Regression

---

## Dataset

IMDb Dataset of 50K Movie Reviews

Dataset Link:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Workflow

1. Load IMDb Dataset
2. Convert sentiment labels into numerical values
3. Split dataset into training and testing sets
4. Apply TF-IDF Vectorization
5. Train Logistic Regression Model
6. Evaluate model performance
7. Predict sentiment for user-entered text

---

## Model Performance

Accuracy: **89.6%**

---

## Sample Output

### Positive Sentiment

Input:
I absolutely loved this movie. The acting was brilliant.

Output:
😊 POSITIVE SENTIMENT

### Negative Sentiment

Input:
This is the worst movie I have ever watched.

Output:
😞 NEGATIVE SENTIMENT

---

## How to Run

1. Download the IMDb Dataset.
2. Place `IMDB Dataset.csv` in the project folder.
3. Install required libraries:

```bash
pip install pandas scikit-learn
```

4. Run the project:

```bash
python main.py
```

---

## Author

Arkajyoti Dutta

Machine Learning & Data Analytics Enthusiast

