import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd


# Load the SMS Spam Collection dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display first 5 records
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Count spam and ham messages
print("\nMessage Counts:")
print(data["label"].value_counts())
print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Messages:")
print(data.duplicated().sum())

# Remove duplicate messages
data = data.drop_duplicates()

# Check dataset after cleaning
print("\nDataset After Cleaning:")
print(data.shape)

print("\nPreprocessing Completed!")
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining and Testing Data Created!")
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Text Vectorization Completed!")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
# Save trained model and vectorizer
import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Both files saved successfully!")

print("Model Training Completed!")

# Step 8: Test Model Accuracy

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\n===== SMART SPAM DETECTION =====")
# Save model performance metrics

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

metrics = {
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"],
    "confusion_matrix": confusion_matrix(
        y_test, y_pred
    )
}

joblib.dump(metrics, "model_metrics.pkl")

print("Model performance metrics saved!")

while True:
    message = input("\nEnter an SMS message (or type exit): ")

    if message.lower() == "exit":
        print("Thank you for using Spam Detection!")
        break

    # Convert message into numerical features
    message_tfidf = vectorizer.transform([message])

    # Predict the message
    prediction = model.predict(message_tfidf)

    if prediction[0] == "spam":
        print("Result: SPAM MESSAGE")
    else:
        print("Result: NOT SPAM (HAM)")
        joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and Vectorizer Saved Successfully!")