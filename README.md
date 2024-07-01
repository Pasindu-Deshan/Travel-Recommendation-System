# Travel-Recommendation-System

The Personalized Travel Recommendation System is a machine learning-based application designed to provide tailored travel destination recommendations to users based on their interests, budget, and preferred climate. The system utilizes a combination of content-based filtering and machine learning classifiers to generate accurate and personalized travel suggestions.

# Key Features
Data Input:

The system reads travel destination data from a CSV file named location_data.csv.
The data includes attributes such as interests, budget, climate, and destination.
Feature Engineering:

The interests, budget, and climate data are combined into a single feature for each destination.
Data Preprocessing:

The combined feature data is split into training and testing sets using an 80-20 split.
The TfidfVectorizer is used to convert the text data into numerical features suitable for machine learning models.
Machine Learning Models:

Random Forest Classifier: A robust ensemble learning method using 200 estimators for classification.
K-Nearest Neighbors (KNN) Classifier: A simple, yet effective classifier using 14 neighbors for prediction.
Model Training and Evaluation:

Both classifiers are trained on the training set and evaluated on the testing set.
Accuracy scores are calculated for both models to assess their performance.
User Input and Validation:

The system prompts the user to input their name, budget, interests, and preferred climate.
Input validation ensures that the budget is a valid integer and the interests and climate match predefined categories.
Content-Based Filtering:

Cosine similarity is used to find the most similar destination based on the user's profile.
The TfidfVectorizer transforms the user profile into numerical features, and similarity scores are calculated against the training data.
Destination Recommendations:

The system predicts destinations using the Random Forest and KNN classifiers.
If all three methods (Random Forest, KNN, and cosine similarity) agree on the destination, that destination is recommended.
If there are discrepancies, multiple destination options are provided to the user.
