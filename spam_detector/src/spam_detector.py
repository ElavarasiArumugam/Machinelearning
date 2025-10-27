import os
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time

# === 1. Download stopwords if needed ===
nltk.download('stopwords')

# === 2. Folder setup ===
output_dir = '../outputs'
results_dir = os.path.join(output_dir, 'results')
models_dir = os.path.join(output_dir, 'models')
data_dir = os.path.abspath('../data')  # folder for datasets

# Create folders if they don't exist
for folder in [results_dir, models_dir, data_dir]:
    os.makedirs(folder, exist_ok=True)
print("Output folders ready!")
print(f"Data folder: {data_dir}")

# === 3. Detect dataset ===
data_path_csv = os.path.join(data_dir, 'spam.csv')
data_path_tsv = os.path.join(data_dir, 'spam.tsv')

if os.path.exists(data_path_csv):
    data_path = data_path_csv
    sep = ','
elif os.path.exists(data_path_tsv):
    data_path = data_path_tsv
    sep = '\t'
else:
    raise FileNotFoundError(
        f"No dataset found in {data_dir}.\n"
        "Please download 'spam.csv' or 'spam.tsv' and place it in the data folder."
    )

# === 4. Load dataset robustly ===
df = pd.read_csv(data_path, sep=sep, encoding='latin-1', engine='python', on_bad_lines='skip')

# Auto-detect label and message columns
if df.shape[1] >= 2:
    df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'message'})
else:
    raise ValueError("Dataset must have at least two columns (label and message).")

# Clean labels and messages
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['message'] = df['message'].astype(str).str.strip()

# Drop empty messages
df = df[df['message'] != '']

# Keep only valid labels
df = df[df['label'].isin(['ham', 'spam'])].reset_index(drop=True)

# Show dataset info
print(f"Cleaned dataset size: {df.shape[0]} messages")
print(df['label'].value_counts())
print(df.head())

if df.shape[0] < 2:
    raise ValueError("Dataset too small to split. Please add more messages.")

# === 5. Preprocessing ===
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_message'] = df['message'].apply(preprocess)
print("Text preprocessing complete!")

# === 6. Feature extraction (TF-IDF) ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label'].map({'ham': 0, 'spam': 1})

# === 7. Split data safely ===
test_size = 0.2 if df.shape[0] >= 5 else 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# === 8. Train Naive Bayes classifier ===
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training complete!")

# === 9. Cross-validation ===
cv_folds = min(5, X_train.shape[0])
cv_accuracy = None
if cv_folds > 1:
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
    cv_accuracy = cv_scores.mean()

# === 10. Evaluate model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

timestamp = int(time.time())
result_filename = f'evaluation_{timestamp}.txt'
with open(os.path.join(results_dir, result_filename), 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    if cv_accuracy:
        f.write(f"Cross-Validation Accuracy: {cv_accuracy:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)
print(f"Evaluation results saved as: {result_filename}")

# === 11. Save model and vectorizer ===
model_filename = f'spam_model_{timestamp}.pkl'
vectorizer_filename = f'vectorizer_{timestamp}.pkl'
joblib.dump(model, os.path.join(models_dir, model_filename))
joblib.dump(vectorizer, os.path.join(models_dir, vectorizer_filename))
print(f"Model saved as: {model_filename}")
print(f"Vectorizer saved as: {vectorizer_filename}")

# === 12. Interactive prediction ===
print("\nPredict a new message")
try:
    new_msg = input("Enter message text: ")
    clean_msg = preprocess(new_msg)
    X_new = vectorizer.transform([clean_msg])
    pred = model.predict(X_new)[0]
    print(f"Predicted label: {'spam' if pred == 1 else 'ham'}")
except ValueError:
    print("Invalid input!")
