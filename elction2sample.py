import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset with generated data
data = {
    'text': [
        "Excited to support Candidate A! #VoteForA",
        "Disagree with Candidate B's policies. #NotForB",
        "Attended a rally for Candidate C. #C2023",
        "Just heard Candidate D's speech. Impressive! #D4President",
        "Not sure about any candidate yet. #Undecided",
        "Supporting Candidate A all the way! #AforPresident",
        "Concerned about Candidate C's stance on key issues. #C2023",
        "Listening to Candidate D's interview. #D4President"
    ],
    'sentiment': ['positive', 'negative', 'positive', 'positive', 'neutral', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)
