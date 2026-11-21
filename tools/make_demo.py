"""Render an annotated loop from the same 3D fixtures used by the unit tests."""

from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from hand_fixture import EDGES, POSES, hand_pose
from recgnize_gesture import recognize_gesture


def font(size, bold=False):
    for path in (
        "malgunbd.ttf" if bold else "malgun.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def draw_frame(name, angle):
    canvas = Image.new("RGB", (640, 360), "#121c29")
    draw = ImageDraw.Draw(canvas)
    points = hand_pose(POSES[name], angle)
    predicted = recognize_gesture(points) or "Other pose"
    draw.text((30, 25), "HAND GESTURE RECOGNITION", fill="#9cb5d0", font=font(16, True))
    draw.text((30, 48), "Synthetic 3D landmark demo", fill="#e9f1fc", font=font(26, True))
    draw.rounded_rectangle((29, 99, 328, 326), radius=15, fill="#1c2d40")
    projected = [(190 + 105 * p.x, 295 - 122 * p.y) for p in points]
    for first, second in EDGES:
        draw.line([projected[first], projected[second]], fill="#7ae4cf", width=4)
    for x, y in projected:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#eef6ff")
    draw.text((355, 130), "Predicted gesture", fill="#9cb5d0", font=font(17))
    draw.text((355, 159), predicted, fill="#7ae4cf", font=font(28, True))
    draw.text((355, 219), f"Hand yaw: {angle:02d}°", fill="#e9f1fc", font=font(20))
    draw.text((355, 260), "Landmark-only check", fill="#9cb5d0", font=font(16))
    return canvas


def main():
    frames = [
        draw_frame(name, angle).resize((512, 288), Image.Resampling.LANCZOS)
        for name in ("Open palm", "Victory", "Thumb", "Closed fist")
        for angle in (0, 15, 30, 45, 60, 80, 60, 30)
    ]
    output = ROOT / "docs" / "landmark-demo.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[0].quantize(colors=24)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=270, loop=0)


if __name__ == "__main__":
    main()
