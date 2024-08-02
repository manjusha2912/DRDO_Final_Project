import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the stopwords module
from nltk.corpus import stopwords 


import warnings

warnings.filterwarnings('ignore')


# Define the path to your CSV file
csv_file_path = 'E:\MANJUSHA\Internships\DRDO\KaggleDatas\Submission\TrainingDataset.csv'

# Read the CSV file using pandas
data = pd.read_csv(csv_file_path, encoding='latin1')

# Extract sentences and labels
sentences = data['Sentence'].values
labels = data['Label'].values

# Split the data into training and testing sets
sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer(min_df = 2, max_df = 0.8, stop_words = stopwords.words('english'))
sentences = vectorizer.fit_transform(sentences.astype('U')).toarray()

# Split the data into training and testing sets
sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
print(sentences_train.shape)
print(labels_train.shape)
print(sentences_test.shape)
print(labels_test.shape)


#LOGISTIC REGRESSION

lr_clf = LogisticRegression()
lr_clf.fit(sentences_train, labels_train)
labels_pred_lr = lr_clf.predict(sentences_test)


# Evaluate the model
accuracy_lr = accuracy_score(labels_test, labels_pred_lr)
report_lr = classification_report(labels_test, labels_pred_lr)
confusion_matrix_lr = confusion_matrix(labels_test, labels_pred_lr)

print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")
print("Classification Report:")
print(report_lr)
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix_lr)


# RANDOM FORREST CLASSIFCATION
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(sentences_train, labels_train)
labels_pred_rf = rf_model.predict(sentences_test)

# Evaluate the model
accuracy_rf = accuracy_score(labels_test, labels_pred_rf)
report_rf = classification_report(labels_test, labels_pred_rf)
confusion_matrix_rf = confusion_matrix(labels_test, labels_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print("Random Forest Classification Report:")
print(report_rf)
print("Random Forest Confusion Matrix:")
print(confusion_matrix_rf)

# ADA BOOST MODEL

ada_model = AdaBoostClassifier(n_estimators=100)
ada_model.fit(sentences_train, labels_train)
labels_pred_ada = ada_model.predict(sentences_test)


# Evaluate the model
accuracy_ada = accuracy_score(labels_test, labels_pred_ada)
report_ada = classification_report(labels_test, labels_pred_ada)
confusion_matrix_ada = confusion_matrix(labels_test, labels_pred_ada)

print(f"ADA Boost Accuracy: {accuracy_ada:.2f}")
print("Classification Report:")
print(report_ada) 
print("ADA Boost Confusion Matrix:")
print(confusion_matrix_ada)


# STEP PREDICTION OF LABELS FOR WEB APPLICATION

# Read the new CSV file
new_csv_file_path = 'E:\MANJUSHA\Internships\DRDO\KaggleDatas\Submission\Offline_forum_test.csv'

new_data_lr = pd.read_csv(new_csv_file_path)
new_data_rf = pd.read_csv(new_csv_file_path)
new_data_ada = pd.read_csv(new_csv_file_path)

# Assuming the new data has a column 'sentence' for which we need to predict the labels
new_sentences = new_data_lr['Sentence'].values

# Vectorize the new sentences using the already fitted vectorizer
sentences_new = vectorizer.transform(new_sentences)

# Predict the labels using the trained model
predicted_labels_lr = lr_clf.predict(sentences_new)
predicted_labels_rf = rf_model.predict(sentences_new)
predicted_labels_ada = ada_model.predict(sentences_new)

# Add the predicted labels to the new data DataFrame
new_data_lr['Label'] = predicted_labels_lr
new_data_rf['Label'] = predicted_labels_rf
new_data_ada['Label'] = predicted_labels_ada

# Save the new data with the predicted labels to a new CSV file

#LR pred
o_csv_file_lr = 'E:\MANJUSHA\Internships\DRDO\KaggleDatas\Submission\logisticR_Pred.csv'
new_data_lr.to_csv(o_csv_file_lr, index=False)

print(f"Predicted labels saved to {o_csv_file_lr}")

# RF pred
o_csv_file_rf = 'E:\MANJUSHA\Internships\DRDO\KaggleDatas\Submission\RForrest_Pred.csv'
new_data_rf.to_csv(o_csv_file_rf, index=False)

print(f"Predicted labels saved to {o_csv_file_rf}")

# ADA Boost pred
o_csv_file_ada = 'E:\MANJUSHA\Internships\DRDO\KaggleDatas\Submission\ADABoost_Pred.csv'
new_data_ada.to_csv(o_csv_file_ada, index=False)

print(f"Predicted labels saved to {o_csv_file_ada}")