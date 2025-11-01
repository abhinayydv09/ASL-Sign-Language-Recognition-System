import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

data_dict = pickle.load(open('data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Ensure 'data' is a 2D NumPy array with consistent shapes
max_num_features = max(len(sample) for sample in data)
data = np.array([sample + [0] * (max_num_features - len(sample)) for sample in data])

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Oversample the underrepresented class
oversampler = RandomOverSampler(random_state=42)
data, labels = oversampler.fit_resample(data, labels)

# Shuffle the data and labels to ensure randomness
data, labels = shuffle(data, labels, random_state=42)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create and train Random Forest classifier
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)

# Evaluate Random Forest model
y_predict_rf = model_rf.predict(x_test)
score_rf = accuracy_score(y_predict_rf, y_test)
cm_rf = confusion_matrix(y_test, y_predict_rf)
print(f'{model_rf} Confusion Matrix:')
print(cm_rf)

# Create and train SVM classifier
model_svm = SVC(kernel='linear')
model_svm.fit(x_train, y_train)

# Evaluate SVM model
y_predict_svm = model_svm.predict(x_test)
score_svm = accuracy_score(y_predict_svm, y_test)
cm_svm = confusion_matrix(y_test, y_predict_svm)
print(f'{model_svm} Confusion Matrix:')
print(cm_svm)

# Create and train Gradient Boosting classifier
model_gb = GradientBoostingClassifier()
model_gb.fit(x_train, y_train)

# Evaluate Gradient Boosting model
y_predict_gb = model_gb.predict(x_test)
score_gb = accuracy_score(y_predict_gb, y_test)
cm_gb = confusion_matrix(y_test, y_predict_gb)
print(f'{model_gb} Confusion Matrix:')
print(cm_gb)

# Plotting accuracy scores
accuracy_scores = [score_rf, score_gb, score_svm]
model_names = ['Random Forest', 'Gradient Boosting', 'Support Vector Machine']

plt.figure(figsize=(10, 5))
plt.bar(model_names, accuracy_scores, color=['blue', 'green', 'red'])
plt.title('Accuracy Scores of Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)  # Set y-axis limit to range from 0 to 1 for accuracy scores
plt.show()

# Plotting confusion matrices
confusion_matrices = [cm_rf, cm_gb, cm_svm]

plt.figure(figsize=(15, 5))
for i, (model_name, cm) in enumerate(zip(model_names, confusion_matrices), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
plt.tight_layout()
plt.show()
