from ppadb.client import Client as AdbClient
import io
from PIL import Image
import cv2
import numpy as np
import os
from datetime import datetime
import time
import platform
from functools import wraps
from logger import Logger
from text_recognition import TextRecognitionSingleton


# Enhanced performance decorator with logger support
def timing_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Use class logger if available, otherwise fall back to print
        if hasattr(self, 'logger') and self.logger:
            self.logger.debug(f"[Performance] {func.__name__:<25} took {execution_time:.4f} seconds")
        else:
            print(f"[Performance] {func.__name__:<25} took {execution_time:.4f} seconds")
            
        return result
    return wrapper

class Actions:
    def __init__(self, device_serial=None):
        # Initialize logger with device serial
        self.logger = Logger(name="Actions", device_serial=device_serial, log_level="INFO")
        self.device_serial = device_serial
        
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_folder = os.path.join(self.script_dir, 'main_images')

        # Initialize ADB connection
        self.adb_client = AdbClient(host="127.0.0.1", port=5037)
        self.device = None
        self._connect_device(device_serial)

        # BlueStacks default resolution (you may need to adjust based on your setup)
        self.device_width = 1080
        self.device_height = 1920
        
        # Define game area coordinates in device space (not screen space)
        # These need to be adjusted based on your BlueStacks resolution and Clash Royale layout
        self.TOP_LEFT_X = 0
        self.TOP_LEFT_Y = 0
        self.BOTTOM_RIGHT_X = 1080
        self.BOTTOM_RIGHT_Y = 1920
        self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)
        
        self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
        self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y
        
        # Card bar coordinates in device space
        self.CARD_BAR_X = 237
        self.CARD_BAR_Y = 1590
        self.CARD_BAR_WIDTH = 820
        self.CARD_BAR_HEIGHT = 250
        
        self.last_screenshot = None

    def _connect_device(self, device_serial=None):
        """Connect to a specific ADB device or the first one available."""
        try:
            devices = self.adb_client.devices()
            if not devices:
                self.logger.error("No ADB devices found. Make sure emulators are running and ADB is enabled.")
                self.logger.warning("Run setup_adb.py first to configure ADB connection.")
                return False
                
            if device_serial:
                self.device = self.adb_client.device(device_serial)
                if not self.device:
                    self.logger.error(f"Device with serial {device_serial} not found.")
                    return False
            else:
                self.device = devices[0]
            
            self.logger.success(f"Connected to device: {self.device.serial}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ADB device: {e}")
            self.logger.warning("Run setup_adb.py first to configure ADB connection.")
            return False

    @timing_decorator
    def _take_screenshot(self):
        """Take a screenshot using ADB"""
        if not self.device:
            self.logger.error("No device connected")
            return None
        
        try:
            screenshot_data = self.device.screencap()
            screenshot = Image.open(io.BytesIO(screenshot_data))
            self.last_screenshot = screenshot.copy()
            return screenshot
        except Exception as e:
            self.logger.error(f"Failed to take screenshot: {e}")
            return None

    def _click(self, x, y):
        """Click at coordinates using ADB"""
        if not self.device:
            self.logger.error("No device connected")
            return False
        
        try:
            self.device.shell(f"input touchscreen swipe {x} {y} {x} {y} 300")
            return True
        except Exception as e:
            self.logger.error(f"Failed to click at ({x}, {y}): {e}")
            return False

    def _swipe(self, x1, y1, x2, y2, duration=500):
        """Swipe from (x1,y1) to (x2,y2) using ADB"""
        if not self.device:
            self.logger.error("No device connected")
            return False
        
        try:
            self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to swipe: {e}")
            return False

    def capture_area(self, save_path, screenshot=None):
        """Capture screenshot of game area using ADB"""
        if screenshot is None:
            screenshot = self._take_screenshot()
        if screenshot:
            # Crop to game area
            cropped = screenshot.crop((self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y))
            cropped.save(save_path)
        else:
            self.logger.error("Failed to capture screenshot")

    def capture_card_area(self, save_path):
        """Capture screenshot of card area using ADB"""
        screenshot = self._take_screenshot()
        if screenshot:
            # Crop to card bar area
            cropped = screenshot.crop((
                self.CARD_BAR_X, 
                self.CARD_BAR_Y, 
                self.CARD_BAR_X + self.CARD_BAR_WIDTH, 
                self.CARD_BAR_Y + self.CARD_BAR_HEIGHT
            ))
            cropped.save(save_path)
        else:
            self.logger.error("Failed to capture card area screenshot")

    @timing_decorator
    def capture_individual_cards(self):
        """Capture and split card bar into individual card images using ADB"""
        screenshot = self._take_screenshot()
        if not screenshot:
            self.logger.error("Failed to capture screenshot for individual cards")
            return []
            
        # Crop to card bar area
        card_bar = screenshot.crop((
            self.CARD_BAR_X, 
            self.CARD_BAR_Y, 
            self.CARD_BAR_X + self.CARD_BAR_WIDTH, 
            self.CARD_BAR_Y + self.CARD_BAR_HEIGHT
        ))
        
        # Calculate individual card widths
        card_width = self.CARD_BAR_WIDTH // 4
        cards = []
        
        # Split into 4 individual card images
        for i in range(4):
            left = i * card_width
            card_img = card_bar.crop((left, 0, left + card_width, self.CARD_BAR_HEIGHT))
            save_path = os.path.join(self.script_dir, 'screenshots', f"{self.device_serial}_card_{i+1}.png")
            card_img.save(save_path)
            cards.append(save_path)

        return cards

    def count_elixir(self):
        """Count elixir using ADB screenshot analysis"""
        screenshot = self._take_screenshot()
            
        if not screenshot:
            self.logger.error("Failed to capture screenshot for elixir counting")
            return 0

        # Define elixir bar region in device coordinates (you may need to adjust these)
        elixir_start_y = 1818  # Approximate Y coordinate of elixir bar
        elixir_end_y = 1890  # Approximate Y coordinate of elixir bar
        elixir_start_x = 280  # Start X coordinate
        elixir_end_x = 360    # End X coordinate


        # Convert PIL image to numpy array for OpenCV processing
        cropped = screenshot.crop((elixir_start_x, elixir_start_y, elixir_end_x, elixir_end_y))
        cropped.save(os.path.join(self.script_dir, 'screenshots', f"{self.device_serial}_elixir_bar.png"))
        screenshot_np = np.array(cropped)
        

        text_recognition = TextRecognitionSingleton()
        result = text_recognition.model.predict(input="screenshots/" + f"{self.device_serial}_elixir_bar.png")

        print(result)
        print(result[0]['rec_text'])
        return int(result[0]['rec_text']) if result and result[0]['rec_text'].isnumeric() else 0

    def _find_template(self, template_path, confidence=0.8, region=None, screenshot=None):
        """Find template image in screenshot using OpenCV template matching"""
        screenshot = screenshot or self._take_screenshot()
        
        if not screenshot:
            return None

        # Convert PIL to OpenCV format
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Crop to region if specified
        if region:
            x, y, w, h = region
            screenshot_cv = screenshot_cv[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            offset_x, offset_y = 0, 0
            
        # Load template
        template = cv2.imread(template_path)
        if template is None:
            self.logger.error(f"Could not load template: {template_path}")
            return None
            
        # Perform template matching
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            # Return center coordinates with offset
            template_h, template_w = template.shape[:2]
            center_x = max_loc[0] + template_w // 2 + offset_x
            center_y = max_loc[1] + template_h // 2 + offset_y
            return (center_x, center_y, max_val)
        
        return None

    def card_play(self, x, y, card_index):
        """Play a card using ADB commands"""
        if card_index in range(4):  # Valid card indices are 0-3
            # Calculate card position in the deck
            card_width = self.CARD_BAR_WIDTH // 4
            card_center_x = self.CARD_BAR_X + (card_index * card_width) + (card_width // 2)
            card_center_y = self.CARD_BAR_Y + (self.CARD_BAR_HEIGHT // 2)

            self.logger.info(f"Clicking on card {card_index} at deck position ({card_center_x}, {card_center_y})")
            self._click(card_center_x, card_center_y)
            time.sleep(0.1)
            
            if y > 1440:
                y = 1440
            if x > 1070:
                x = 1070

            self.logger.info(f"Placing card at battlefield position ({x}, {y})")
            self._click(x, y)
        else:
            self.logger.error(f"Invalid card index: {card_index} (must be 0-3)")

    def click_battle_start(self):
        """Find and click the battle start button using ADB and template matching"""
        button_image = os.path.join(self.images_folder, "battlestartbutton.png")
        confidences = [0.8, 0.7, 0.6, 0.5]  # Try multiple confidence levels

        # Define the region for the battle button in device coordinates
        battle_button_region = (300, 1350, 480, 300)  # Adjust based on your device resolution

        while True:
            for confidence in confidences:
                self.logger.debug(f"Looking for battle start button (confidence: {confidence})")
                result = self._find_template(button_image, confidence, battle_button_region)
                if result:
                    x, y, match_confidence = result
                    self.logger.info(f"Found battle button at ({x}, {y}) with confidence {match_confidence}")
                    self._click(x, y)
                    time.sleep(2)
                    return

            # If button not found, click to clear screens
            self.logger.info("Button not found, clicking to clear screens...")
            self._click(640, 200)  # Center-ish click in device coordinates
            
            # Check for popups and click OK if found
            self.click_ok_button()

    def click_ok_button(self):
        """Click the OK button to close popups"""
        ok_button_image = os.path.join(self.images_folder, "okbutton.png")
        confidences = [0.8]

        # Define the region for the OK button in device coordinates
        ok_button_region = (0, 0, 1080, 1920)  # Adjust based on your device resolution
        for confidence in confidences:
            self.logger.debug(f"Looking for OK button (confidence: {confidence})")
            result = self._find_template(ok_button_image, confidence, ok_button_region)
            if result:
                x, y, match_confidence = result
                self.logger.info(f"Found OK button at ({x}, {y}) with confidence {match_confidence}")
                self._click(x, y)
                time.sleep(2)
                return

    def detect_game_end(self):
        """Detect game end using ADB and template matching"""
        try:
            winner_img = os.path.join(self.images_folder, "Winner.png")
            confidences = [0.8, 0.7, 0.6]

            # Define winner detection region in device coordinates
            winner_region = (0, 0, 1080, 1920)  # Adjust based on your device resolution

            for confidence in confidences:
                # print(f"\nTrying detection with confidence: {confidence}")

                result = self._find_template(winner_img, confidence, winner_region, screenshot=self.last_screenshot)
                if result:
                    x, y, match_confidence = result
                    self.logger.debug(f"Found 'Winner' at ({x}, {y}) with confidence {match_confidence}")
                    
                    # Determine if victory or defeat based on position
                    return "victory" if y > 300 else "defeat"
        except Exception as e:
            self.logger.error(f"Error in game end detection: {str(e)}")
        return None

    def detect_match_over(self):
        """Detect match over using ADB and template matching"""
        matchover_img = os.path.join(self.images_folder, "matchover.png")
        confidences = [0.8, 0.6, 0.4]
        
        # Define the region where the matchover image appears in device coordinates
        region = (100, 570, 900, 200)  # Adjust based on your device resolution
        
        for confidence in confidences:
            result = self._find_template(matchover_img, confidence, region, screenshot=self.last_screenshot)
            if result:
                self.logger.info("Match over detected!")
                return True
                
        return False
