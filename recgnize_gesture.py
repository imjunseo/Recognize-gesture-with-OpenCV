"""Classify hand gestures from MediaPipe's 3D landmarks."""

import argparse
from math import dist, isfinite
from pathlib import Path


FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
GESTURES = {
    (True, True, True, True, True): "Open palm",
    (False, True, True, False, False): "Victory",
    (False, False, True, False, False): "Middle finger",
    (False, False, False, False, True): "Pinky",
    (True, False, False, False, False): "Thumb",
    (False, True, False, False, False): "Pointing up",
    (True, True, False, False, False): "Pointing up",
    (False, False, False, False, False): "Closed fist",
}


def straightness(points):
    length = sum(dist(first, second) for first, second in zip(points, points[1:]))
    return dist(points[0], points[-1]) / length if length else 0.0


def recognize_gesture(landmarks):
    points = [(point.x, point.y, point.z) for point in landmarks]
    if len(points) != 21 or not all(isfinite(value) for point in points for value in point):
        return None
    palm_width = dist(points[5], points[17])
    if palm_width <= 1e-9:
        return None
    wrist = points[0]
    thumb = straightness(points[1:5]) >= 0.93 and dist(points[4], points[5]) >= 0.55 * palm_width
    fingers = tuple(
        straightness([points[index] for index in joints]) >= 0.88
        and dist(wrist, points[joints[-1]]) >= 1.08 * dist(wrist, points[joints[1]])
        for joints in FINGERS
    )
    return GESTURES.get((thumb, *fingers))


def inspect_frame(cv2, mp, tracker, frame):
    result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width = frame.shape[:2]
    labels = []
    for image_hand, world_hand in zip(
        result.multi_hand_landmarks or (), result.multi_hand_world_landmarks or ()
    ):
        label = recognize_gesture(world_hand.landmark) or "Other pose"
        labels.append(label)
        wrist = image_hand.landmark[0]
        position = (max(0, int(wrist.x * width) - 45), max(24, int(wrist.y * height) - 24))
        cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, image_hand, mp.solutions.hands.HAND_CONNECTIONS
        )
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--video", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--no-display", action="store_true", help="Print image results without a window.")
    args = parser.parse_args()
    if args.no_display and args.image is None:
        parser.error("--no-display requires --image")

    import cv2
    import mediapipe as mp

    with mp.solutions.hands.Hands(static_image_mode=args.image is not None) as tracker:
        if args.image is not None:
            frame = cv2.imread(str(args.image))
            if frame is None:
                parser.error(f"Unable to read image: {args.image}")
            print(", ".join(inspect_frame(cv2, mp, tracker, frame)) or "No hand detected")
            if not args.no_display:
                try:
                    cv2.imshow("Gesture Recognition", frame)
                    cv2.waitKey(0)
                finally:
                    cv2.destroyAllWindows()
            return

        description = str(args.video) if args.video else f"webcam {args.camera}"
        capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        try:
            if not capture.isOpened():
                parser.error(f"Unable to open {description}")
            while True:
                success, frame = capture.read()
                if not success:
                    if args.video:
                        break
                    parser.error(f"Unable to read a frame from {description}")
                inspect_frame(cv2, mp, tracker, frame)
                cv2.imshow("Gesture Recognition", frame)
                if (cv2.waitKey(1 if not args.video else 30) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
