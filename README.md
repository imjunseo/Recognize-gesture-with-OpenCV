# 웹캠 손 제스처 인식

OpenCV로 웹캠 영상을 읽고 MediaPipe가 추정한 손 랜드마크를 이용해 손 모양을 분류합니다. 펼친 손, V, 가운데손가락, 엄지 제스처를 손가락 관절과 손목 사이의 거리 관계로 구분하고 화면에 표시합니다.

## 실행

Python 3.11과 웹캠이 필요합니다. 가상 환경을 활성화한 뒤 저장소 루트에서 실행하세요.

```bash
python -m pip install -r requirements.txt
python recgnize_gesture.py
```

영상 창에서 `Q` 또는 `Esc`로 종료합니다. 손이 옆으로 돌아가거나 일부 손가락이 가려지면 거리 규칙에 따라 오인식될 수 있습니다.

- [인식 코드](recgnize_gesture.py)
- [MediaPipe 손 추적 참고 문서](https://google.github.io/mediapipe/solutions/hands.html)
