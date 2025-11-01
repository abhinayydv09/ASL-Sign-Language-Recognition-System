#gradient boosting algorithm training
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
# Load data
data_dict = pickle.load(open('data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Ensure 'data' is a 2D NumPy array with consistent shapes
max_num_features = max(len(sample) for sample in data)
data = np.array([sample + [0] * (max_num_features - len(sample)) for sample in data])

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Shuffle the data and labels to ensure randomness
data, labels = shuffle(data, labels, random_state=42)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create Gradient Boosting classifier
model = GradientBoostingClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

#print('Accuracy:', accuracy)

cm = confusion_matrix(y_test, y_pred)
print(f'{model} Confusion Matrix:')
print(cm)

with open('model3.p', 'wb') as f:
    pickle.dump({'model': model}, f)
