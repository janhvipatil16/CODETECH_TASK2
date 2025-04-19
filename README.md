# CODETECH_TASK2

# Sentiment Analysis on Customer Reviews Using TF-IDF Vectorization and Logistic Regression

This project performs sentiment analysis on a dataset of customer reviews using Natural Language Processing (NLP) techniques. The task involves analyzing the sentiment of movie reviews from the IMDB dataset, where the sentiment is classified as either "positive" or "negative."

The analysis is carried out using **TF-IDF (Term Frequency-Inverse Document Frequency)** for text vectorization and **Logistic Regression** for classification. This method allows us to efficiently process text data, train a model, and evaluate the model's performance.

## Project Overview

1. **Dataset**: IMDB Dataset of Movie Reviews (CSV format), which contains two columns: `text` (review text) and `label` (sentiment label - "positive" or "negative").

2. **Goal**: Build a machine learning model to predict the sentiment of a given movie review (positive or negative) based on the content of the review text.

3. **Key Steps**:
   - Preprocess the review text data.
   - Perform TF-IDF vectorization on the text data.
   - Train a Logistic Regression model to classify the sentiment.
   - Evaluate model performance using a confusion matrix and classification report.

## Steps

### 1. Import Libraries
We begin by importing all necessary libraries, including:
- **Pandas**: For data handling.
- **NumPy**: For numerical operations.
- **re** and **string**: For text cleaning.
- **scikit-learn**: For machine learning model training and evaluation.
- **Matplotlib** and **Seaborn**: For visualizing model performance.

### 2. Load Dataset
The dataset is loaded from a local CSV file, which contains movie reviews with associated sentiment labels. We rename the columns for better readability and map the sentiment labels to numerical values (`positive` -> 1, `negative` -> 0).

### 3. Balance Data
We ensure that the dataset is balanced by equalizing the number of positive and negative reviews. We randomly sample from each class to achieve the same number of samples for both classes.

### 4. Clean Text
The text data is preprocessed by:
- Converting text to lowercase.
- Removing any HTML tags.
- Removing non-alphabetical characters.
- Removing extra whitespace.

The cleaned text is stored in a new column, `cleaned_text`.

### 5. TF-IDF Vectorization
We use the **TF-IDF Vectorizer** from scikit-learn to transform the cleaned text into numerical features. We limit the number of features to 5000 to avoid overfitting and maintain computational efficiency.

### 6. Train/Test Split
The data is split into training and testing sets using **train_test_split** from scikit-learn, with 80% of the data used for training and 20% for testing. The split ensures that the model is trained on one subset and evaluated on another.

### 7. Train the Logistic Regression Model
We train a **Logistic Regression** model on the training data. The `max_iter` parameter is set to 1000 to ensure the model converges during training.

### 8. Model Evaluation
We evaluate the performance of the trained model by:
- Generating a **classification report**, which includes precision, recall, F1-score, and accuracy.
- Visualizing the **confusion matrix** using **Seaborn** to display how well the model performs on the test data.

### Example Output
The output includes:
- A **classification report** displaying precision, recall, F1-score, and accuracy for both classes.
- A **confusion matrix** heatmap that visually represents the model's predictions.

## Results

After running the model, you will see the following:
- A classification report that shows the precision, recall, F1-score, and accuracy for both positive and negative sentiment classes.
- A confusion matrix heatmap that visualizes the performance of the model in terms of true positives, true negatives, false positives, and false negatives.

## Conclusion

This project demonstrates how to perform sentiment analysis using text data and machine learning techniques. By leveraging TF-IDF for text vectorization and Logistic Regression for classification, we can efficiently analyze customer reviews and classify sentiments as either positive or negative.
