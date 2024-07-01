import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Read data from the CSV file
df = pd.read_csv('location_data.csv')

# Combine interests, climate and budget into a single feature
df['features'] = df['interests'] + ' ' + df['budget'].astype(str) + ' ' + df['climate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['features'], df['destination'], test_size=0.2, random_state=62)

# Use TF-IDF vectorizer to convert text data into numerical features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=60)
rf_classifier.fit(X_train_tfidf, y_train)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=14)
knn_classifier.fit(X_train_tfidf, y_train)

# Make predictions with random forest
rf_predictions = rf_classifier.predict(X_test_tfidf)

# Make predictions with KNN
knn_predictions = knn_classifier.predict(X_test_tfidf)

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Example: Predict destination for a new user
user_name = input("Enter your name : ")
user_budget = input("Enter your budget : $ ")

# Check if user input for budget is a valid integer
try:
    user_budget = int(user_budget)
    user_interests = input("Enter your interests ('beach', 'mountain', 'historical', 'adventure', 'city') : ").lower()
    user_climate = input("Enter your preferred climate ('tropical', 'temperate', 'arctic', 'desert') : ").lower()

    # Check if user input is one of the specified interests and climate
    allowed_interests = ['beach', 'mountain', 'historical', 'adventure', 'city']
    allowed_climates = ['tropical', 'temperate', 'arctic', 'desert']

    if user_climate not in allowed_climates:
        print("Error: Invalid climate preference. Please enter one of the following: 'tropical', 'temperate', 'arctic', 'desert'")
    elif user_interests not in allowed_interests:
        print("Error: Invalid interest. Please enter one of the following interests: 'beach', 'mountain', 'historical', 'adventure', 'city'")
    else:
        # Create a dictionary for the new user
        new_user = {'budget': user_budget, 'interests': user_interests, 'preferred_climate': user_climate}
        new_user_features = new_user['interests'] + ' ' + str(new_user['budget']) + ' ' + new_user['preferred_climate']

        # Content-Based Filtering: Use cosine similarity to find similar destinations
        user_profile_tfidf = vectorizer.transform([new_user_features])

        # Calculate cosine similarity between the new user's profile and existing destinations
        similarity_scores = cosine_similarity(user_profile_tfidf, X_train_tfidf)

        # Find the index of the most similar destination
        most_similar_index = np.argmax(similarity_scores)

        # Get the corresponding destination
        cosine_similar_destination = y_train.iloc[most_similar_index]


        # Use models to predict destination for the new user
        new_user_tfidf = vectorizer.transform([new_user_features])
        rf_prediction = rf_classifier.predict(new_user_tfidf)[0]
        knn_prediction = knn_classifier.predict(new_user_tfidf)[0]

        if rf_prediction == knn_prediction == cosine_similar_destination:
          print(f'Hello {user_name}, we recommend you to travel to the {rf_prediction}')
        elif rf_prediction == knn_prediction and rf_prediction != cosine_similar_destination:
          print(f'Hello {user_name}, we recommend you to travel to the {rf_prediction} or {cosine_similar_destination}')
        elif rf_prediction == knn_prediction and knn_prediction != cosine_similar_destination:
          print(f'Hello {user_name}, we recommend you to travel to the {knn_prediction} or {cosine_similar_destination}')
        else:
          print(f'Hello {user_name}, we recommend you to travel to the {rf_prediction}, {knn_prediction}, or {cosine_similar_destination}')


except ValueError:
    print("Error: Invalid budget. Please enter a valid integer.")
    exit()