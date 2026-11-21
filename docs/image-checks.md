# 실제 이미지 확인

MediaPipe Gesture Recognizer 공개 예제 사진을 내려받아 아래 명령을 실행했습니다. 사진 원본은 이 저장소에 포함하지 않았습니다.

| 이미지 | 출력 |
| --- | --- |
| [thumbs_up.jpg](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg) | Thumb |
| [victory.jpg](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/victory.jpg) | Victory |
| [pointing_up.jpg](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg) | Pointing up |

```bash
python recgnize_gesture.py --image victory.jpg --no-display
```

이 사진들은 구현 과정에서 수동으로 확인한 표본입니다. 방향별 실사용 정확도를 별도로 평가하려면 같은 제스처를 정면, 45도, 측면에서 촬영해 사진별 결과와 검출 실패 건수를 기록하세요.
