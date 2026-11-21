import unittest

from hand_fixture import POSES, hand_pose
from recgnize_gesture import recognize_gesture


class GestureTests(unittest.TestCase):
    def test_supported_gestures_from_front_and_side(self):
        for label, fingers in POSES.items():
            for angle in (0, 45, 80):
                with self.subTest(gesture=label, angle=angle):
                    self.assertEqual(recognize_gesture(hand_pose(fingers, angle)), label)

    def test_unsupported_or_unusable_pose(self):
        landmarks = hand_pose((0, 1, 0, 1, 0))
        self.assertIsNone(recognize_gesture(landmarks))
        self.assertIsNone(recognize_gesture(landmarks[:-1]))
        landmarks[0].z = float("nan")
        self.assertIsNone(recognize_gesture(landmarks))


if __name__ == "__main__":
    unittest.main()
