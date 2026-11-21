import cv2
import mediapipe as mp


FINGER_JOINTS = ((18, 4), (6, 8), (10, 12), (14, 16), (18, 20))
GESTURES = {
    (True, True, True, True, True): "Open palm",
    (False, True, True, False, False): "Victory",
    (False, False, True, False, False): "Middle finger",
    (False, False, False, False, True): "Pinky",
    (True, False, False, False, False): "Thumb",
}


def distance(first, second):
    return abs(first.x - second.x) + abs(first.y - second.y)


def recognize_gesture(landmarks):
    wrist = landmarks[0]
    extended = tuple(
        distance(wrist, landmarks[joint]) < distance(wrist, landmarks[tip])
        for joint, tip in FINGER_JOINTS
    )
    return GESTURES.get(extended)


def main():
    camera = cv2.VideoCapture(0)
    hands = mp.solutions.hands
    try:
        if not camera.isOpened():
            raise SystemExit("Unable to open webcam 0.")

        with hands.Hands() as tracker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise SystemExit("Unable to read a frame from webcam 0.")

                height, width = frame.shape[:2]
                result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for landmarks in result.multi_hand_landmarks or ():
                    label = recognize_gesture(landmarks.landmark)
                    if label is not None:
                        wrist = landmarks.landmark[0]
                        position = (max(0, int(wrist.x * width) - 50),
                                    max(25, int(wrist.y * height) - 30))
                        cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 255, 255), 2)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, hands.HAND_CONNECTIONS
                    )

                cv2.imshow("Gesture Recognition", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
