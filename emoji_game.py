import os
import random
import base64
import time
import cv2
import ollama
import json
import numpy as np
from enum import Enum

# --- Performance Parameters ------------------------
os.environ.setdefault("OLLAMA_NUM_THREAD", "8")

# Dual resolution setup
WEBCAM_WIDTH, WEBCAM_HEIGHT = 640, 480   # Webcam capture resolution
RESIZE_FOR_MODEL = (224, 224)            # Low res for AI
JPEG_QUALITY = 70

FRAME_SKIP = 3  # Check AI every 3 frames

# --- Game Parameters ------------------------------
EMOJIS = ["😀", "😢", "😮", "😡", "😎", "😜", "😴", "😋", "🤔", "😱"]
MODEL = "qwen2.5vl:3b"
TIMEOUT = 15  # seconds per emoji
CAM_ID = 0
TOTAL_ROUNDS = 5

# --- UI Layout Parameters -------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WEBCAM_DISPLAY_WIDTH = 320
WEBCAM_DISPLAY_HEIGHT = 240
SNAPSHOT_WIDTH = 320
SNAPSHOT_HEIGHT = 240

# --- UI Colors -----------------------------------
class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)  
    ORANGE = (255, 165, 0)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)
    CYAN = (0, 255, 255)


class GameState(Enum):
    MENU = 1
    PLAYING = 2
    ROUND_END = 3
    GAME_OVER = 4

class EmojiGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        
        self.state = GameState.MENU
        self.score = 0
        self.current_round = 0
        self.target_emoji = ""
        self.start_time = 0
        self.frame_count = 0
        self.last_proximity = 0
        self.proximity_history = []
        self.best_score = 0
        self.round_completed = False
        self.current_feedback = "Welcome!"
        self.last_snapshot = None
        self.round_success = False
        
        # Create window
        cv2.namedWindow("Emoji Challenge", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Emoji Challenge", SCREEN_WIDTH, SCREEN_HEIGHT)
        
    def encode_b64(self, frame):
        """Encode frame for AI processing (low resolution)"""
        small = cv2.resize(frame, RESIZE_FOR_MODEL)
        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(buf).decode()
    
    def get_proximity_feedback(self, img_b64: str, target: str) -> dict:
        """Get detailed feedback about expression accuracy"""
        prompt = (
            f"Analyze how well the facial expression matches the emoji {target}.\n"
            f"Rate the similarity from 0-100 and provide brief feedback.\n"
            f"Respond in JSON: {{\"similarity\": <0-100>, \"feedback\": \"<brief tip>\", \"match\": <true/false>}}"
        )
        
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64]
                }],
                options={"temperature": 0.1}
            )
            
            content = response['message']['content']
            print(f"AI Response: {content}")  # Debug output
            
            # Try to extract JSON from response
            if '{' in content and '}' in content:
                json_str = content[content.find('{'):content.rfind('}')+1]
                result = json.loads(json_str)
                return {
                    "similarity": min(100, max(0, result.get("similarity", 0))),
                    "feedback": result.get("feedback", "Keep trying!"),
                    "match": result.get("match", False)
                }
        except Exception as e:
            print(f"AI Error: {e}")
        
        # Fallback to simple check
        try:
            match = '"match":true' in response['message']['content'].lower() or 'true' in response['message']['content'].lower()
            return {
                "similarity": 85 if match else random.randint(20, 60),
                "feedback": "Perfect!" if match else "Try adjusting your expression",
                "match": match
            }
        except:
            return {
                "similarity": 30,
                "feedback": "Keep trying!",
                "match": False
            }
    
    def create_game_screen(self):
        """Create the main game screen background"""
        # Create a dark background
        screen = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 40, dtype=np.uint8)
        
        # Add gradient effect
        for y in range(SCREEN_HEIGHT):
            intensity = int(40 + (y / SCREEN_HEIGHT) * 20)
            screen[y, :] = [intensity, intensity, intensity]
        
        return screen
    
    def draw_webcam_feed(self, screen, webcam_frame):
        """Draw webcam feed in the right corner"""
        # Resize webcam frame
        webcam_resized = cv2.resize(webcam_frame, (WEBCAM_DISPLAY_WIDTH, WEBCAM_DISPLAY_HEIGHT))
        
        # Position in right corner
        x_pos = SCREEN_WIDTH - WEBCAM_DISPLAY_WIDTH - 20
        y_pos = 100
        
        # Add border
        cv2.rectangle(screen, (x_pos-3, y_pos-3), 
                     (x_pos + WEBCAM_DISPLAY_WIDTH + 3, y_pos + WEBCAM_DISPLAY_HEIGHT + 3), 
                     Colors.WHITE, 2)
        
        # Place webcam feed
        screen[y_pos:y_pos + WEBCAM_DISPLAY_HEIGHT, 
               x_pos:x_pos + WEBCAM_DISPLAY_WIDTH] = webcam_resized
        
        # Label
        cv2.putText(screen, "LIVE", (x_pos, y_pos - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.GREEN, 2)
    
    def draw_snapshot(self, screen):
        """Draw last round snapshot in the left corner"""
        x_pos = 20
        y_pos = 100
        
        # Draw border
        cv2.rectangle(screen, (x_pos-3, y_pos-3), 
                     (x_pos + SNAPSHOT_WIDTH + 3, y_pos + SNAPSHOT_HEIGHT + 3), 
                     Colors.WHITE, 2)
        
        if self.last_snapshot is not None:
            # Resize and place snapshot
            snapshot_resized = cv2.resize(self.last_snapshot, (SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT))
            screen[y_pos:y_pos + SNAPSHOT_HEIGHT, 
                   x_pos:x_pos + SNAPSHOT_WIDTH] = snapshot_resized
            
            # Label with result
            result_text = "SUCCESS!" if self.round_success else "FAILED"
            result_color = Colors.GREEN if self.round_success else Colors.RED
            cv2.putText(screen, result_text, (x_pos, y_pos - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
        else:
            # Empty placeholder
            cv2.rectangle(screen, (x_pos, y_pos), 
                         (x_pos + SNAPSHOT_WIDTH, y_pos + SNAPSHOT_HEIGHT), 
                         Colors.DARK_GRAY, -1)
            cv2.putText(screen, "LAST ROUND", (x_pos + 80, y_pos + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.GRAY, 2)
    
    def draw_top_ui(self, screen):
        """Draw top UI elements"""
        # Round counter
        round_text = f"ROUND {self.current_round}/{TOTAL_ROUNDS}"
        cv2.putText(screen, round_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, Colors.WHITE, 2)
        
        # Score
        score_text = f"SCORE: {self.score}"
        cv2.putText(screen, score_text, (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, Colors.GREEN, 2)
        
        # Timer
        if self.state == GameState.PLAYING:
            elapsed = time.time() - self.start_time
            remaining = max(0, TIMEOUT - elapsed)
            timer_text = f"TIME: {remaining:.1f}s"
            timer_color = Colors.RED if remaining < 5 else Colors.WHITE
            cv2.putText(screen, timer_text, (SCREEN_WIDTH - 200, 50), 
                       cv2.FONT_HERSHEY_COMPLEX, 1.2, timer_color, 2)
    
    def draw_center_ui(self, screen):
        """Draw center UI elements"""
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        if self.state == GameState.PLAYING:
            # Target emoji (large)
            emoji_text = f"MATCH THIS: {self.target_emoji}"
            text_size = cv2.getTextSize(emoji_text, cv2.FONT_HERSHEY_COMPLEX, 2, 3)[0]
            text_x = center_x - text_size[0] // 2
            cv2.putText(screen, emoji_text, (text_x, center_y - 100), 
                       cv2.FONT_HERSHEY_COMPLEX, 2, Colors.YELLOW, 3)
            
            # Very large emoji display
            large_emoji = self.target_emoji * 3  # Make it bigger visually
            emoji_size = cv2.getTextSize(large_emoji, cv2.FONT_HERSHEY_COMPLEX, 4, 5)[0]
            emoji_x = center_x - emoji_size[0] // 2
            cv2.putText(screen, large_emoji, (emoji_x, center_y), 
                       cv2.FONT_HERSHEY_COMPLEX, 4, Colors.YELLOW, 5)
    
    def draw_bottom_ui(self, screen):
        """Draw bottom UI elements"""
        bottom_y = SCREEN_HEIGHT - 80
        
        # Proximity bar
        if self.proximity_history and self.state == GameState.PLAYING:
            proximity = self.proximity_history[-1]
            bar_width = 400
            bar_height = 30
            bar_x = (SCREEN_WIDTH - bar_width) // 2
            
            # Background bar
            cv2.rectangle(screen, (bar_x, bottom_y), (bar_x + bar_width, bottom_y + bar_height), 
                         Colors.DARK_GRAY, -1)
            
            # Progress bar
            progress_width = int(bar_width * proximity / 100)
            if proximity > 80:
                bar_color = Colors.GREEN
            elif proximity > 50:
                bar_color = Colors.YELLOW
            else:
                bar_color = Colors.RED
            
            cv2.rectangle(screen, (bar_x, bottom_y), (bar_x + progress_width, bottom_y + bar_height), 
                         bar_color, -1)
            
            # Border
            cv2.rectangle(screen, (bar_x, bottom_y), (bar_x + bar_width, bottom_y + bar_height), 
                         Colors.WHITE, 2)
            
            # Proximity text
            prox_text = f"ACCURACY: {proximity}%"
            text_size = cv2.getTextSize(prox_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (SCREEN_WIDTH - text_size[0]) // 2
            cv2.putText(screen, prox_text, (text_x, bottom_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, Colors.WHITE, 2)
        
        # Feedback text
        if hasattr(self, 'current_feedback'):
            feedback_size = cv2.getTextSize(self.current_feedback, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            feedback_x = (SCREEN_WIDTH - feedback_size[0]) // 2
            cv2.putText(screen, self.current_feedback, (feedback_x, SCREEN_HEIGHT - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, Colors.CYAN, 2)
    
    def draw_menu(self, screen):
        """Draw the main menu"""
        center_x = SCREEN_WIDTH // 2
        
        # Title
        title = "EMOJI CHALLENGE"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_COMPLEX, 3, 4)[0]
        title_x = center_x - title_size[0] // 2
        cv2.putText(screen, title, (title_x, 150), cv2.FONT_HERSHEY_COMPLEX, 3, Colors.YELLOW, 4)
        
        # Instructions
        instructions = [
            "Match your facial expression to the emoji!",
            f"Complete {TOTAL_ROUNDS} rounds to win",
            "",
            "PRESS SPACE TO START",
            "PRESS ESC TO QUIT"
        ]
        
        start_y = 250
        for i, line in enumerate(instructions):
            if "PRESS" in line:
                color = Colors.GREEN
                font_scale = 1.2
                thickness = 3
            else:
                color = Colors.WHITE
                font_scale = 1
                thickness = 2
            
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            cv2.putText(screen, line, (text_x, start_y + i * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Best score
        if self.best_score > 0:
            best_text = f"BEST SCORE: {self.best_score}/{TOTAL_ROUNDS}"
            best_size = cv2.getTextSize(best_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            best_x = center_x - best_size[0] // 2
            cv2.putText(screen, best_text, (best_x, SCREEN_HEIGHT - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, Colors.ORANGE, 2)
    
    def draw_game_over(self, screen):
        """Draw game over screen"""
        center_x = SCREEN_WIDTH // 2
        
        # Title
        title = "GAME OVER!"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_COMPLEX, 3, 4)[0]
        title_x = center_x - title_size[0] // 2
        cv2.putText(screen, title, (title_x, 200), cv2.FONT_HERSHEY_COMPLEX, 3, Colors.RED, 4)
        
        # Final score
        score_text = f"FINAL SCORE: {self.score}/{TOTAL_ROUNDS}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_COMPLEX, 2, 3)[0]
        score_x = center_x - score_size[0] // 2
        
        if self.score >= TOTAL_ROUNDS * 0.8:
            score_color = Colors.GREEN
        elif self.score >= TOTAL_ROUNDS * 0.5:
            score_color = Colors.YELLOW
        else:
            score_color = Colors.RED
            
        cv2.putText(screen, score_text, (score_x, 300), cv2.FONT_HERSHEY_COMPLEX, 2, score_color, 3)
        
        # Performance message
        if self.score == TOTAL_ROUNDS:
            message = "PERFECT! 🏆"
        elif self.score >= TOTAL_ROUNDS * 0.8:
            message = "EXCELLENT! 🌟"
        elif self.score >= TOTAL_ROUNDS * 0.6:
            message = "GOOD JOB! 👍"
        else:
            message = "KEEP PRACTICING! 💪"
        
        msg_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_COMPLEX, 1.5, 2)[0]
        msg_x = center_x - msg_size[0] // 2
        cv2.putText(screen, message, (msg_x, 370), cv2.FONT_HERSHEY_COMPLEX, 1.5, Colors.CYAN, 2)
        
        # Options
        options = ["SPACE - Play Again", "ESC - Quit"]
        start_y = 450
        for i, option in enumerate(options):
            opt_size = cv2.getTextSize(option, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            opt_x = center_x - opt_size[0] // 2
            cv2.putText(screen, option, (opt_x, start_y + i * 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, Colors.WHITE, 2)
    
    def start_new_round(self):
        """Start a new round"""
        self.current_round += 1
        self.target_emoji = random.choice(EMOJIS)
        self.start_time = time.time()
        self.frame_count = 0
        self.proximity_history = []
        self.round_completed = False
        self.current_feedback = "Get ready to match the emoji!"
        self.state = GameState.PLAYING
        print(f"Round {self.current_round}: Match {self.target_emoji}")
    
    def end_round(self, success, webcam_frame):
        """End current round"""
        self.round_success = success
        self.last_snapshot = webcam_frame.copy()
        if success:
            self.score += 1
            self.current_feedback = "SUCCESS! Great job! 🎉"
        else:
            self.current_feedback = "Time's up! Try again next round!"
        
        print(f"Round {self.current_round} ended: {'SUCCESS' if success else 'FAILED'}")
        
        # Brief pause before continuing
        time.sleep(2)
        
        if self.current_round >= TOTAL_ROUNDS:
            self.state = GameState.GAME_OVER
        else:
            self.start_new_round()
    
    def reset_game(self):
        """Reset the game"""
        if self.score > self.best_score:
            self.best_score = self.score
        self.score = 0
        self.current_round = 0
        self.last_snapshot = None
        self.state = GameState.MENU
    
    def run(self):
        """Main game loop"""
        print("🎮 Enhanced Emoji Challenge Game Started!")
        print("Controls: SPACE = Start/Continue, ESC = Quit")
        
        while True:
            ret, webcam_frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam!")
                break
            
            # Flip webcam frame for mirror effect
            webcam_frame = cv2.flip(webcam_frame, 1)
            
            # Create main game screen
            screen = self.create_game_screen()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key
                break
            
            # State machine
            if self.state == GameState.MENU:
                self.draw_menu(screen)
                if key == 32:  # SPACE key
                    self.start_new_round()
            
            elif self.state == GameState.PLAYING:
                # Draw all UI elements
                self.draw_top_ui(screen)
                self.draw_center_ui(screen)
                self.draw_bottom_ui(screen)
                self.draw_webcam_feed(screen, webcam_frame)
                self.draw_snapshot(screen)
                
                # Check if time is up
                elapsed = time.time() - self.start_time
                if elapsed >= TIMEOUT and not self.round_completed:
                    self.end_round(False, webcam_frame)
                    continue
                
                # Process frame with AI (every FRAME_SKIP frames)
                if self.frame_count % FRAME_SKIP == 0 and not self.round_completed:
                    try:
                        img_b64 = self.encode_b64(webcam_frame)
                        feedback = self.get_proximity_feedback(img_b64, self.target_emoji)
                        
                        self.proximity_history.append(feedback["similarity"])
                        self.current_feedback = feedback["feedback"]
                        
                        # Keep only recent history
                        if len(self.proximity_history) > 10:
                            self.proximity_history.pop(0)
                        
                        # Check for success
                        if feedback["match"]:
                            self.round_completed = True
                            self.end_round(True, webcam_frame)
                    
                    except Exception as e:
                        print(f"AI processing error: {e}")
                        self.current_feedback = "AI processing error - continue playing"
                
                self.frame_count += 1
            
            elif self.state == GameState.GAME_OVER:
                self.draw_game_over(screen)
                self.draw_snapshot(screen)  # Show last snapshot
                if key == 32:  # SPACE key
                    self.reset_game()
            
            # Show the screen
            cv2.imshow("Emoji Challenge", screen)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n🎯 Game ended! Best score: {self.best_score}/{TOTAL_ROUNDS}")

def main():
    """Main function"""
    try:
        # Test if Ollama is available
        ollama.list()
        print("✅ Ollama connection successful!")
        
        game = EmojiGame()
        game.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running and the model is available!")
        print("You can start Ollama with: ollama serve")

if __name__ == "__main__":
    main()