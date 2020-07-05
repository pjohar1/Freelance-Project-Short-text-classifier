# Freelance-Project-Short-text-classifier
Classifier to predict the merchant category of the business entity from a credit card transaction

## Data Collection:
We scrape Google search results for the business part in the credit card transaction.

## Feature Extraction:
All the data from the Google search result is stored in a JSON file and we extract those features from it.

## Data Exploration:
We analyze which features are more interesting and revealing for our project

## Modeling:
We take 2 apporaches:
Sparse Model: We convert the text into tokens and get the TF-IDF of the words. Then use SVM algorithm to classify the merchant category( average accuracy: 75.2%)
Word Embeddings: Use Neural Network to train the classifier. (aaverage accuracy: 74.9%)

## Drawbacks:
- We have 7013 observations and 15 categories, which means we have too little data.
- Data is also imbalanced as few categories have very less data as compared to others.

## Way forward:
- Collect more data for categories with less data.
- Try using pre-trained word embeddings like Glove for training the classifier.
