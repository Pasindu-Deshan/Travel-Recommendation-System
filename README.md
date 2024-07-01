# Travel-Recommendation-System

The Personalized Travel Recommendation System is a machine learning-based application designed to provide tailored travel destination recommendations to users based on their interests, budget, and preferred climate. The system utilizes a combination of content-based filtering and machine-learning classifiers to generate accurate and personalized travel suggestions.

# Key Features
Data Input:<br />
<li>The system reads travel destination data from a CSV file named location_data.csv.</li>
<li>The data includes interests, budget, climate, and destination attributes.</li><br />

Feature Engineering:<br />
<li>The interests, budget, and climate data are combined into a single feature for each destination.</li><br />

Data Preprocessing:<br />
<li>The combined feature data is split into training and testing sets using an 80-20 split.</li>
<li>The TfidfVectorizer converts the text data into numerical features suitable for machine learning models.</li><br />

Machine Learning Models:<br />
<li>Random Forest Classifier: A robust ensemble learning method using 200 estimators for classification.</li>
<li>K-Nearest Neighbors (KNN) Classifier: A simple, yet effective classifier using 14 neighbors for prediction.</li><br />

Model Training and Evaluation:<br />
<li>Both classifiers are trained on the training set and evaluated on the testing set.</li>
<li>Accuracy scores are calculated for both models to assess their performance.</li><br />

User Input and Validation:<br />
<li>The system prompts users to input their name, budget, interests, and preferred climate.</li>
<li>Input validation ensures that the budget is a valid integer and that the interests and climate match predefined categories.</li><br />

Content-Based Filtering:<br />
<li>Cosine similarity is used to find the most similar destination based on the user's profile.</li>
<li>The TfidfVectorizer transforms the user profile into numerical features, and similarity scores are calculated against the training data.</li><br />

Destination Recommendations:<br />
<li>The system predicts destinations using the Random Forest and KNN classifiers.</li>
<li>If all three methods (Random Forest, KNN, and cosine similarity) agree on the destination, that destination is recommended.</li>
<li>If there are discrepancies, multiple destination options are provided to the user.</li><br />
