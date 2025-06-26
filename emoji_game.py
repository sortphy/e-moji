import os
import random
import base64
import time
import cv2
import ollama

# --- Parâmetros de performance ------------------------
# Limita as threads usadas pelo Ollama ‒ ajuste conforme seus núcleos físicos
os.environ.setdefault("OLLAMA_NUM_THREAD", "8")

# Reduz resolução e compressão da imagem enviada ao modelo
FRAME_WIDTH, FRAME_HEIGHT = 320, 240   # captura da webcam
RESIZE_FOR_MODEL = (224, 224)           # resol. enviada ao LLM
JPEG_QUALITY = 70                       # 0‑100, menor = +compressão

# Envia menos frames para o modelo (checa 1 a cada N)
FRAME_SKIP = 5

# --- Parâmetros de jogo ------------------------------
EMOJIS  = ["😀","😢","😮","😡","😎","😜"]

# MODEL  = "qwen2.5vl:7b"    # MODELO PESADO  
MODEL = "qwen2.5vl:3b"        # MODELO LEVE

TIMEOUT = 10                     # segundos por emoji
CAM_ID  = 0                      # webcam padrão

# -----------------------------------------------------

def encode_b64(frame):
    """Reduz resolução, comprime e retorna base64."""
    small = cv2.resize(frame, RESIZE_FOR_MODEL)
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode()


def check_expression(img_b64: str, target: str) -> bool:
    """Envia o frame ao Qwen2.5‑VL e retorna True se bate com o emoji alvo."""
    prompt = (
        f"Verifique se a expressão facial imita *exatamente* o emoji {target}.\n"
        "Responda apenas em JSON: {\"match\": true|false} "
    )

    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_b64]
        }],
        # zero randomness → ligeiramente mais rápido
        options={"temperature": 0}
    )

    return '"match":true' in response['message']['content'].lower()


def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    score = 0
    for round_n in range(1, len(EMOJIS) + 1):
        target = random.choice(EMOJIS)
        start  = time.time()
        frame_count = 0

        print(f"\nRound {round_n}: imite {target}")

        while time.time() - start < TIMEOUT:
            ok, frame = cap.read()
            if not ok:
                continue

            cv2.putText(frame, f"Imite {target}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Emoji Game", frame)

            # Envia ao modelo só a cada FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0:
                if check_expression(encode_b64(frame), target):
                    print("✅ Acertou!")
                    score += 1
                    break
            frame_count += 1

            # ESC encerra
            if cv2.waitKey(1) == 27:
                print("\nJogo encerrado pelo usuário.")
                cap.release()
                cv2.destroyAllWindows()
                return
        else:
            print("⏰ Tempo esgotado!")

    print(f"\nPontuação final: {score}/{round_n}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
