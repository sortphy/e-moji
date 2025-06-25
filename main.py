import cv2
import random
import numpy as np
from pathlib import Path
from deepface import DeepFace
from PIL import Image

# --- Config --------------------------------------------------
EMOJI_DIR = Path(__file__).with_suffix("").parent / "emojis"
EMOJI_MAP = {
    "happy": EMOJI_DIR / "happy.png",
    "angry": EMOJI_DIR / "angry.png",
    "surprise": EMOJI_DIR / "surprise.png",
    "sad": EMOJI_DIR / "sad.png",
}
FPS_TARGET = 24  # display fps
ANALYZE_EVERY_N_FRAMES = FPS_TARGET // 2  # ~2× per second
EMOJI_SIZE = 120  # px square

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


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No webcam found.")
        return

    current_emoji_name, current_emoji_img = choose_random_emoji()
    score = 0
    frame_id = 0

    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        frame = overlay_emoji(frame, current_emoji_img)

        # Analyze expression every N frames
        if frame_id % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                dominant = result[0]["dominant_emotion"] if isinstance(result, list) else result["dominant_emotion"]
                if dominant == current_emoji_name:
                    score += 1
                    current_emoji_name, current_emoji_img = choose_random_emoji()
            except Exception as e:
                # DeepFace can fail on bad frames; ignore
                print("Analyze error:", e)

        # HUD
        cv2.putText(frame, f"Target: {current_emoji_name}", (10, EMOJI_SIZE + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score}", (10, EMOJI_SIZE + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Emoji Face Game", frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
