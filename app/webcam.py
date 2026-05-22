import cv2

from src.config import DEVICE, IMG_SIZE
from src.predict import predict_image
from app.inference import initialize_model, draw_emotion_on_frame


class WebcamProcessor:
    def __init__(self, model_path='models/best_model.pth'):
        self.model = initialize_model(model_path)
        self.device = DEVICE
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            label = "No face detected"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            prediction = predict_image(self.model, face_resized, self.device)
            frame = draw_emotion_on_frame(frame, prediction, (x, y, w, h))

        return frame

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            frame = self.process_frame(frame)
            cv2.imshow('Facial Emotion Recognition - Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def get_webcam_processor(model_path='models/best_model.pth'):
    return WebcamProcessor(model_path)
