from flask import Flask, request, jsonify
import cv2
from deepface import DeepFace
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

@app.route('/')
def index():
    return "Hello from Flask!"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'succes': 'working'}), 200

#generate try catch for handling errors in python


@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        print("request.files['image'] ", str(request.files))
        # Check if image is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read image file from request
        image = request.files['image'].read()
        print('Image received. Size:', len(image))  # Add this line for debugging

        # Convert image data to numpy array
        nparr = np.frombuffer(image, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces in the image
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(img, 1.3, 5)
        
        emotions = []
        for (x,y,w,h) in faces:
            # Crop face from image
            face_img = img[y:y+h, x:x+w]
            # Predict emotion
            emotion_result = DeepFace.analyze(img_path=face_img, actions=['emotion', 'age'], enforce_detection=False)
            print('Emotion analysis result:', emotion_result)  # Add this line for debugging
            if 'dominant_emotion' in emotion_result[0]:
                emotion = emotion_result[0]
                emotions.append(emotion)
            else:
                # Handle case where no face is detected
                emotions.append('No face detected')
        return jsonify({'emotions': emotions, 'age': emotion_result[0]['age']}), 200
    except Exception as e:
        # Handle any other exception
        print('Error occurred:', str(e))  # Add this line for debugging
        return jsonify({'error': str(e)}), 500  # Change status code to 500 for server error

if __name__ == '__main__':
    app.run()


# ngrok http --domain=oarfish-obliging-rooster.ngrok-free.app 5000