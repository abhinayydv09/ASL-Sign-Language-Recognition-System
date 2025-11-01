#webpage integration
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response
import mediapipe as mp
from gtts import gTTS
import gtts
import pyttsx3
import threading
from playsound import playsound
app = Flask(__name__)

# Load the model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'thumbs up'}

#engine = pyttsx3.init()

def play_audio(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while True:
        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                x_ = []  # Initialize x_ and y_ lists within the loop
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data_aux.extend([0.0] * (84 - len(data_aux)))  # Ensure data has 84 features

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Start a separate thread for continuous audio output
                #threading.Thread(target=play_audio, args=(predicted_character,)).start()
                play_audio(predicted_character)
                
            

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0),
                            3, cv2.LINE_AA)
                
                #print(predicted_character)
                #tts = gtts.gTTS(predicted_character)
                #tts.save("hello.mp3")
                #playsound("hello.mp3")
                #engine = pyttsx3.init()
                

        

        cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    return(predicted_character)
@app.route('/')
def index():
    return render_template('indexweb3.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
