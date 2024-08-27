# Fake Product Review Detection

## Overview

This Streamlit application classifies product reviews as either fake (CG - Computer Generated) or genuine (OR - Original Review) using a Logistic Regression model. The project demonstrates the use of natural language processing (NLP) techniques and machine learning to solve the problem of identifying fake reviews, which is a significant challenge in online marketplaces.

## Features

- **Input Review Classification**: Enter a product review in the app, and it will classify the review as fake (CG) or genuine (OR).
- **Model Accuracy Display**: The application shows the accuracy of the trained Logistic Regression model.
- **Real-time Classification**: The app provides instant feedback on the classification of the entered review.

## Technologies Used

- **Python**: The core programming language.
- **Streamlit**: For building the web interface.
- **scikit-learn**: For machine learning models and preprocessing.
- **pandas**: For data manipulation.
- **TfidfVectorizer**: For text vectorization.

## Dataset

The dataset used in this project contains product reviews labeled as either fake (CG) or genuine (OR). It is included in this repository as `fake_reviews_dataset.csv`. The dataset has been preprocessed to map labels to binary values for classification.

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed. If you do not have Python installed, you can download it from [python.org](https://www.python.org/).



