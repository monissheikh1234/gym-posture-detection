from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for counters and stages
counters = {"armcurl": 0, "pushup": 0, "weightlifting": 0, "squat": 0, "lunge": 0, "plank": 0}
stages = {"armcurl": None, "pushup": None, "weightlifting": None, "squat": None, "lunge": None, "plank": None}
feedback = ""  # Store feedback
selected_exercise = None

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180 else angle

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('Contact.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')
@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/exercise/<exercise_name>')
def exercise(exercise_name):
    global selected_exercise
    selected_exercise = exercise_name
    return render_template('exercise.html', exercise=exercise_name)

def generate_frames():
    cap = cv2.VideoCapture(0)
    global feedback, selected_exercise

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the image for pose detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract necessary landmarks
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle_elbow = calculate_angle(shoulder, elbow, wrist)
                angle_knee = calculate_angle(hip, knee, ankle)
                angle_hip = calculate_angle(shoulder, hip, knee)

                # Exercise-specific logic
                if selected_exercise == "armcurl":
                    if angle_elbow > 160:
                        stages[selected_exercise] = "down"
                    if angle_elbow < 30 and stages[selected_exercise] == "down":
                        stages[selected_exercise] = "up"
                        counters[selected_exercise] += 1
                    feedback = "Align elbow with shoulder!" if abs(elbow[0] - shoulder[0]) > 0.1 else ""

                elif selected_exercise == "pushup":
                    if angle_elbow > 160:
                        stages[selected_exercise] = "down"
                    if angle_elbow < 90 and stages[selected_exercise] == "down":
                        stages[selected_exercise] = "up"
                        counters[selected_exercise] += 1
                    feedback = "Keep body straight!" if abs(hip[1] - shoulder[1]) > 0.1 else ""

                elif selected_exercise == "squat":
                    if angle_knee > 160:
                        stages[selected_exercise] = "up"
                    if angle_knee < 90 and stages[selected_exercise] == "up":
                        stages[selected_exercise] = "down"
                        counters[selected_exercise] += 1
                    feedback = "Lower down until knees are at 90°!" if angle_knee > 90 else ""

                elif selected_exercise == "lunge":
                    if angle_knee > 160:
                        stages[selected_exercise] = "up"
                    if angle_knee < 90 and stages[selected_exercise] == "up":
                        stages[selected_exercise] = "down"
                        counters[selected_exercise] += 1
                    feedback = "Front knee should align over the ankle!" if abs(knee[0] - ankle[0]) > 0.1 else ""

                elif selected_exercise == "plank":
                    feedback = "Keep a straight line from shoulders to ankles!" if abs(hip[1] - shoulder[1]) > 0.1 else "Good form!"

                # Display counter and feedback
                cv2.putText(image, f"Reps: {counters[selected_exercise]}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, feedback, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(e)

            # Draw pose landmarks on image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
