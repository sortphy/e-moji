"""
Emoji Face‑Matching Game (DeepFace Edition)
==========================================

Stack:
  * OpenCV         -> webcam + drawing
  * DeepFace       -> pretrained emotion / age / gender
  * Pillow (PIL)   -> overlay PNG emojis with alpha

Install:
  pip install opencv-python-headless deepface Pillow numpy

Run:
  python emoji_face_game.py

Folder layout:
  project/
    emoji_face_game.py   # (este arquivo)
    emojis/
      happy.png
      angry.png
      sad.png
      surprise.png
"""

import cv2
import random
from pathlib import Path
from PIL import Image
import numpy as np
from deepface import DeepFace

# --- Config --------------------------------------------------
EMOJI_DIR = Path(__file__).with_suffix("").parent / "emojis"
EMOJI_MAP = {
    "happy": EMOJI_DIR / "happy.png",
    "angry": EMOJI_DIR / "angry.png",
    "sad": EMOJI_DIR / "sad.png",
    "surprise": EMOJI_DIR / "surprise.png",
}
EMOJI_SIZE = 120
ANALYZE_EVERY_N_FRAMES = 12   # ~2×/s if camera ~24 fps

# -------------------------------------------------------------

def load_emoji(name: str) -> Image.Image:
    """Load and resize PNG emoji by emotion name."""
    img = Image.open(EMOJI_MAP[name]).convert("RGBA")
    return img.resize((EMOJI_SIZE, EMOJI_SIZE), Image.LANCZOS)


def overlay_emoji(frame: np.ndarray, emoji: Image.Image, x: int = 10, y: int = 10) -> np.ndarray:
    """Overlay RGBA Pillow image onto BGR OpenCV frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    pil_frame.paste(emoji, (x, y), emoji)
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


def choose_random_emoji():
    name = random.choice(list(EMOJI_MAP.keys()))
    return name, load_emoji(name)

# -------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No webcam found.")
        return

    detector_backend = "opencv"  # fastest; alternatives: mtcnn, retinaface …

    current_emoji_name, current_emoji_img = choose_random_emoji()
    score = 0
    frame_id = 0

    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view

        # Analyze expression every N frames
        if frame_id % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion", "age", "gender"],
                    detector_backend=detector_backend,
                    enforce_detection=False,
                )
                # DeepFace 2024+ returns dict. Older versions return list[dict]
                if isinstance(result, list):
                    result = result[0]

                dominant_emotion = result.get("dominant_emotion", "unknown")
                age = result.get("age", "?")
                gender = result.get("gender", "?")

                if dominant_emotion == current_emoji_name:
                    score += 1
                    current_emoji_name, current_emoji_img = choose_random_emoji()
            except Exception as e:
                print("Analyze error:", e)
                dominant_emotion = "err"
                age = gender = "?"
        else:
            dominant_emotion = "..."
            age = gender = "..."

        # HUD
        frame = overlay_emoji(frame, current_emoji_img)
        cv2.putText(frame, f"Target: {current_emoji_name}", (10, EMOJI_SIZE + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score}", (10, EMOJI_SIZE + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {dominant_emotion}", (10, EMOJI_SIZE + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Age: {age}  Gender: {gender}", (10, EMOJI_SIZE + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Emoji Game (DeepFace)", frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
