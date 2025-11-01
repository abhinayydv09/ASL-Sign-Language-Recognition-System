import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load your model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

import string

# Define labels_dict for 26 alphabet letters
labels_dict = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

while True:
    data_aux = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Append the x and y coordinates to data_aux
                data_aux.append(x)
                data_aux.append(y)

        # Ensure that data_aux has 84 features (add zeros to match the model's expected features)
        while len(data_aux) < 84:
            data_aux.extend([0, 0])

        # Make a prediction using the adjusted data
        prediction = model.predict([data_aux])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
