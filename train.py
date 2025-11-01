#random forest traing
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

#print('{}% of samples were classified correctly!'.format(score * 100))

cm = confusion_matrix(y_test, y_predict)
print(f'{model} Confusion Matrix:')
print(cm)

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
