import cv2
import random
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
import pygame
import threading
from datetime import datetime

# --- Enhanced Config ---
EMOJI_DIR = Path(__file__).with_suffix("").parent / "emojis"
EMOJI_MAP = {
    "happy": EMOJI_DIR / "happy.png",
    "angry": EMOJI_DIR / "angry.png", 
    "sad": EMOJI_DIR / "sad.png",
    "surprise": EMOJI_DIR / "surprise.png",
}

# Game settings
MAX_ROUNDS = 10
ROUND_TIME_LIMIT = 15  # seconds per round
EMOJI_SIZE = 150
ANALYZE_EVERY_N_FRAMES = 8
POINTS_PER_CORRECT = 100
TIME_BONUS_MULTIPLIER = 10

# Visual effects settings
PARTICLE_COUNT = 50
FLASH_DURATION = 30  # frames
SCREEN_SHAKE_INTENSITY = 15

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-8, -2)
        self.color = color
        self.life = 60
        self.max_life = 60
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2  # gravity
        self.life -= 1
        
    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / self.max_life
            size = int(alpha * 8)
            if size > 0:
                cv2.circle(frame, (int(self.x), int(self.y)), size, self.color, -1)

class EmojiGame:
    def __init__(self):
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Game state
        self.state = "MENU"  # MENU, PLAYING, GAME_OVER, PAUSED
        self.round_num = 0
        self.score = 0
        self.total_time = 0
        self.round_start_time = 0
        self.current_emoji_name = ""
        self.current_emoji_img = None
        self.detector_backend = "opencv"
        
        # Visual effects
        self.particles = []
        self.flash_timer = 0
        self.shake_timer = 0
        self.last_correct_snapshot = None
        self.show_snapshot = False
        
        # Load emojis
        self.emojis = {}
        self.load_emojis()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception("No webcam found!")
        
        # Set camera to fullscreen resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.frame_id = 0
        
    def load_emojis(self):
        """Load and resize all emoji images"""
        for name, path in EMOJI_MAP.items():
            if path.exists():
                img = Image.open(path).convert("RGBA")
                self.emojis[name] = img.resize((EMOJI_SIZE, EMOJI_SIZE), Image.LANCZOS)
            else:
                # Create placeholder emoji if file doesn't exist
                self.emojis[name] = self.create_placeholder_emoji(name)
    
    def create_placeholder_emoji(self, emotion):
        """Create a simple colored circle as placeholder emoji"""
        colors = {
            "happy": (255, 255, 0, 255),    # Yellow
            "sad": (0, 0, 255, 255),        # Blue  
            "angry": (255, 0, 0, 255),      # Red
            "surprise": (255, 165, 0, 255)  # Orange
        }
        
        img = Image.new("RGBA", (EMOJI_SIZE, EMOJI_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = colors.get(emotion, (128, 128, 128, 255))
        draw.ellipse([10, 10, EMOJI_SIZE-10, EMOJI_SIZE-10], fill=color)
        
        # Add simple face
        if emotion == "happy":
            # Smile
            draw.ellipse([40, 45, 55, 60], fill=(0, 0, 0, 255))  # Left eye
            draw.ellipse([95, 45, 110, 60], fill=(0, 0, 0, 255))  # Right eye
            draw.arc([50, 70, 100, 100], 0, 180, fill=(0, 0, 0, 255))  # Smile
        elif emotion == "sad":
            # Frown
            draw.ellipse([40, 45, 55, 60], fill=(0, 0, 0, 255))
            draw.ellipse([95, 45, 110, 60], fill=(0, 0, 0, 255))
            draw.arc([50, 85, 100, 115], 180, 360, fill=(0, 0, 0, 255))  # Frown
        elif emotion == "angry":
            # Angry eyes and frown
            draw.polygon([(35, 40), (60, 55), (35, 65)], fill=(0, 0, 0, 255))  # Left angry eye
            draw.polygon([(115, 40), (90, 55), (115, 65)], fill=(0, 0, 0, 255))  # Right angry eye
            draw.rectangle([45, 90, 105, 95], fill=(0, 0, 0, 255))  # Angry mouth
        elif emotion == "surprise":
            # Wide eyes and open mouth
            draw.ellipse([35, 40, 60, 65], fill=(0, 0, 0, 255))  # Left wide eye
            draw.ellipse([90, 40, 115, 65], fill=(0, 0, 0, 255))  # Right wide eye
            draw.ellipse([65, 85, 85, 105], fill=(0, 0, 0, 255))  # Open mouth
            
        return img
    
    def choose_random_emoji(self):
        """Select a random emoji for the next round"""
        name = random.choice(list(self.emojis.keys()))
        return name, self.emojis[name]
    
    def start_new_round(self):
        """Initialize a new game round"""
        self.round_num += 1
        self.current_emoji_name, self.current_emoji_img = self.choose_random_emoji()
        self.round_start_time = time.time()
        self.show_snapshot = bool(self.last_correct_snapshot)
        
    def overlay_image(self, frame, img, x, y):
        """Overlay RGBA PIL image onto BGR OpenCV frame"""
        if img.mode != "RGBA":
            img = img.convert("RGBA")
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        
        # Create a copy for blending
        overlay = Image.new("RGBA", pil_frame.size, (0, 0, 0, 0))
        overlay.paste(img, (x, y))
        
        # Blend with original frame
        pil_frame = pil_frame.convert("RGBA")
        blended = Image.alpha_composite(pil_frame, overlay)
        
        return cv2.cvtColor(np.array(blended.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    def add_particles(self, x, y, color=(0, 255, 255)):
        """Add celebration particles at position"""
        for _ in range(PARTICLE_COUNT):
            self.particles.append(Particle(x, y, color))
    
    def update_particles(self, frame):
        """Update and draw all particles"""
        self.particles = [p for p in self.particles if p.life > 0]
        for particle in self.particles:
            particle.update()
            particle.draw(frame)
    
    def apply_screen_shake(self, frame):
        """Apply screen shake effect"""
        if self.shake_timer > 0:
            shake_x = random.randint(-SCREEN_SHAKE_INTENSITY, SCREEN_SHAKE_INTENSITY)
            shake_y = random.randint(-SCREEN_SHAKE_INTENSITY, SCREEN_SHAKE_INTENSITY)
            
            h, w = frame.shape[:2]
            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
            frame = cv2.warpAffine(frame, M, (w, h))
            self.shake_timer -= 1
            
        return frame
    
    def apply_flash_effect(self, frame):
        """Apply flash effect when correct answer"""
        if self.flash_timer > 0:
            intensity = self.flash_timer / FLASH_DURATION
            flash_overlay = np.full_like(frame, (255, 255, 255), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 1 - intensity * 0.5, flash_overlay, intensity * 0.5, 0)
            self.flash_timer -= 1
            
        return frame
    
    def draw_menu(self, frame):
        """Draw the main menu"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Title
        title = "EMOJI EXPRESSION GAME"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2, 3)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, h//3), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        
        # Instructions
        instructions = [
            "Match your facial expression to the emoji!",
            f"• {MAX_ROUNDS} rounds, {ROUND_TIME_LIMIT} seconds each",
            f"• {POINTS_PER_CORRECT} points per correct match",
            f"• Time bonus: {TIME_BONUS_MULTIPLIER} points per second remaining",
            "",
            "Controls:",
            "SPACE - Start Game",
            "P - Pause/Resume",
            "ESC - Quit",
            "",
            "Press SPACE to begin!"
        ]
        
        start_y = h//2
        for i, line in enumerate(instructions):
            if line:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = (w - text_size[0]) // 2
                color = (0, 255, 0) if "Press SPACE" in line else (255, 255, 255)
                cv2.putText(frame, line, (text_x, start_y + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def draw_game_over(self, frame):
        """Draw game over screen"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Game Over title
        title = "GAME OVER!"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2, 3)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, h//3), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        
        # Final stats
        accuracy = (self.score // POINTS_PER_CORRECT) / MAX_ROUNDS * 100
        avg_time = self.total_time / MAX_ROUNDS
        
        stats = [
            f"Final Score: {self.score}",
            f"Accuracy: {accuracy:.1f}%",
            f"Average Time: {avg_time:.1f}s",
            "",
            "Press SPACE to play again",
            "Press ESC to quit"
        ]
        
        start_y = h//2
        for i, line in enumerate(stats):
            if line:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                color = (0, 255, 255) if "Score:" in line else (255, 255, 255)
                cv2.putText(frame, line, (text_x, start_y + i * 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    def draw_hud(self, frame):
        """Draw heads-up display during gameplay"""
        h, w = frame.shape[:2]
        
        # Progress bar background
        bar_width = w - 40
        bar_height = 20
        bar_x, bar_y = 20, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar fill
        progress = self.round_num / MAX_ROUNDS
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Round info
        round_text = f"Round {self.round_num}/{MAX_ROUNDS}"
        cv2.putText(frame, round_text, (bar_x, bar_y + bar_height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Timer
        elapsed = time.time() - self.round_start_time
        remaining = max(0, ROUND_TIME_LIMIT - elapsed)
        timer_color = (0, 0, 255) if remaining < 5 else (255, 255, 255)
        timer_text = f"Time: {remaining:.1f}s"
        cv2.putText(frame, timer_text, (w - 200, bar_y + bar_height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
        
        # Score
        score_text = f"Score: {self.score}"
        cv2.putText(frame, score_text, (bar_x, h - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Target emoji and instruction
        emoji_y = bar_y + 70
        frame = self.overlay_image(frame, self.current_emoji_img, bar_x, emoji_y)
        
        target_text = f"Show: {self.current_emoji_name.upper()}"
        cv2.putText(frame, target_text, (bar_x + EMOJI_SIZE + 20, emoji_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Show last correct snapshot if available
        if self.show_snapshot and self.last_correct_snapshot is not None:
            snapshot_size = 120
            snapshot_x = w - snapshot_size - 20
            snapshot_y = emoji_y
            
            # Resize snapshot
            snapshot_resized = cv2.resize(self.last_correct_snapshot, (snapshot_size, snapshot_size))
            
            # Add border
            cv2.rectangle(frame, (snapshot_x - 2, snapshot_y - 2), 
                         (snapshot_x + snapshot_size + 2, snapshot_y + snapshot_size + 2), 
                         (0, 255, 0), 2)
            
            # Overlay snapshot
            frame[snapshot_y:snapshot_y + snapshot_size, snapshot_x:snapshot_x + snapshot_size] = snapshot_resized
            
            cv2.putText(frame, "Last Success!", (snapshot_x - 20, snapshot_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def check_time_limit(self):
        """Check if round time limit exceeded"""
        elapsed = time.time() - self.round_start_time
        return elapsed >= ROUND_TIME_LIMIT
    
    def handle_correct_answer(self, frame):
        """Handle when player gets correct emotion"""
        # Calculate time bonus
        elapsed = time.time() - self.round_start_time
        time_remaining = max(0, ROUND_TIME_LIMIT - elapsed)
        time_bonus = int(time_remaining * TIME_BONUS_MULTIPLIER)
        
        # Add points
        total_points = POINTS_PER_CORRECT + time_bonus
        self.score += total_points
        self.total_time += elapsed
        
        # Take snapshot
        h, w = frame.shape[:2]
        face_region = frame[h//4:3*h//4, w//4:3*w//4]  # Center portion
        self.last_correct_snapshot = face_region.copy()
        
        # Visual effects
        self.flash_timer = FLASH_DURATION
        self.shake_timer = 10
        self.add_particles(w//2, h//2, (0, 255, 0))
        
        # Start next round or end game
        if self.round_num >= MAX_ROUNDS:
            self.state = "GAME_OVER"
        else:
            self.start_new_round()
    
    def reset_game(self):
        """Reset game state for new game"""
        self.round_num = 0
        self.score = 0
        self.total_time = 0
        self.particles = []
        self.flash_timer = 0
        self.shake_timer = 0
        self.last_correct_snapshot = None
        self.show_snapshot = False
        
    def run(self):
        """Main game loop"""
        # Create fullscreen window
        cv2.namedWindow("Emoji Expression Game", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Emoji Expression Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("=== EMOJI EXPRESSION GAME ===")
        print("Controls:")
        print("SPACE - Start game / Restart")
        print("P - Pause/Resume")
        print("ESC - Quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror view
            
            # Handle different game states
            if self.state == "MENU":
                frame = self.draw_menu(frame)
                
            elif self.state == "PLAYING":
                # Check time limit
                if self.check_time_limit():
                    if self.round_num >= MAX_ROUNDS:
                        self.state = "GAME_OVER"
                    else:
                        elapsed = time.time() - self.round_start_time
                        self.total_time += elapsed
                        self.start_new_round()
                
                # Analyze emotion every N frames
                if self.frame_id % ANALYZE_EVERY_N_FRAMES == 0:
                    try:
                        result = DeepFace.analyze(
                            frame,
                            actions=["emotion"],
                            detector_backend=self.detector_backend,
                            enforce_detection=False,
                        )
                        if isinstance(result, list):
                            result = result[0]
                        
                        dominant_emotion = result.get("dominant_emotion", "unknown")
                        
                        # Check for correct emotion
                        if dominant_emotion == self.current_emoji_name:
                            self.handle_correct_answer(frame)
                            
                    except Exception as e:
                        print(f"Analysis error: {e}")
                
                # Draw game HUD
                frame = self.draw_hud(frame)
                
            elif self.state == "GAME_OVER":
                frame = self.draw_game_over(frame)
            
            # Apply visual effects
            self.update_particles(frame)
            frame = self.apply_flash_effect(frame)
            frame = self.apply_screen_shake(frame)
            
            cv2.imshow("Emoji Expression Game", frame)
            self.frame_id += 1
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                if self.state in ["MENU", "GAME_OVER"]:
                    self.reset_game()
                    self.state = "PLAYING"
                    self.start_new_round()
            elif key == ord('p') or key == ord('P'):  # PAUSE
                if self.state == "PLAYING":
                    self.state = "PAUSED"
                elif self.state == "PAUSED":
                    self.state = "PLAYING"
        
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def main():
    try:
        game = EmojiGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required libraries installed:")
        print("pip install opencv-python pillow numpy deepface pygame")

if __name__ == "__main__":
    main()