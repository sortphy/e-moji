

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import mediapipe as mp
import random

# --- Config --------------------------------------------------
EMOJI_DIR = Path(__file__).with_suffix("").parent / "emojis"
EMOJI_MAP = {
    "happy": EMOJI_DIR / "happy.png",
    "angry": EMOJI_DIR / "angry.png",
    "sad": EMOJI_DIR / "sad.png",
    "surprise": EMOJI_DIR / "surprise.png",
}
EMOJI_SIZE = 120
FPS_TARGET = 24
CHECK_EVERY_N_FRAMES = FPS_TARGET // 2

mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# --- Helpers ------------------------------------------------

def load_emoji(name):
    img = Image.open(EMOJI_MAP[name]).convert("RGBA")
    return img.resize((EMOJI_SIZE, EMOJI_SIZE))

def overlay_emoji(frame, emoji, x=10, y=10):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    pil_frame.paste(emoji, (x, y), emoji)
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- Feature Checks ------------------------------------------

def extract_landmarks(results, frame_shape):
    if not results.multi_face_landmarks:
        return None
    h, w, _ = frame_shape
    landmarks = [(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark]
    return landmarks

def is_happy(landmarks):
    return distance(landmarks[61], landmarks[291]) > 60  # mouth corners wide

def is_angry(landmarks):
    brow = distance(landmarks[70], landmarks[107]) + distance(landmarks[336], landmarks[336 - 37])
    return brow < 25

def is_sad(landmarks):
    return distance(landmarks[66], landmarks[296]) < 40  # mouth compressed

def is_surprise(landmarks):
    return distance(landmarks[13], landmarks[14]) > 25  # mouth open

EXPRESSION_CHECKS = {
    "happy": is_happy,
    "angry": is_angry,
    "sad": is_sad,
    "surprise": is_surprise,
}

# -------------------------------------------------------------

def choose_random_expression():
    name = random.choice(list(EMOJI_MAP.keys()))
    return name, load_emoji(name)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No webcam found.")
        return

    current_expr, current_emoji = choose_random_expression()
    score = 0
    frame_id = 0

    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = FACE_MESH.process(rgb)

        landmarks = extract_landmarks(results, frame.shape)

        if frame_id % CHECK_EVERY_N_FRAMES == 0 and landmarks is not None:
            check_fn = EXPRESSION_CHECKS[current_expr]
            if check_fn(landmarks):
                score += 1
                current_expr, current_emoji = choose_random_expression()

        frame = overlay_emoji(frame, current_emoji)
        cv2.putText(frame, f"Target: {current_expr}", (10, EMOJI_SIZE + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Score: {score}", (10, EMOJI_SIZE + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Emoji Game", frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()