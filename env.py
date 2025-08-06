import numpy as np
import time
import os
import json
import threading
from functools import wraps
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient
from logger import Logger

# Load environment variables from .env file
load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10
NUM_CARD_SLOTS = 4

SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]


from functools import wraps

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

class ClashRoyaleEnv:
    def __init__(self, device_serial=None):
        # Initialize logger with device serial
        self.logger = Logger(name="ClashRoyaleEnv", device_serial=device_serial, log_level="INFO")
        self.device_serial = device_serial
        
        self.actions = Actions(device_serial=device_serial)
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        
        self._load_card_data()

        # state_size = 1 (elixir) + 4*2 (card_id, elixir_cost) + 10*2 (allies) + 10*2 (enemies)
        self.state_size = 1 + (NUM_CARD_SLOTS * 2) + (2 * MAX_ALLIES) + (2 * MAX_ENEMIES)
        
        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28

        self.screenshot_path = os.path.join(os.path.dirname(__file__), 'screenshots', f"{self.device_serial}_current.png")
        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None

        self.prev_enemy_princess_towers = None

    def _load_card_data(self):
        card_data_path = os.path.join(os.path.dirname(__file__), 'cards.json')
        with open(card_data_path, 'r') as f:
            self.card_data = json.load(f)
        
        self.card_to_id = {name: i for i, name in enumerate(self.card_data.keys())}
        self.id_to_card = {i: name for name, i in self.card_to_id.items()}

    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def reset(self):
        self.actions.click_battle_start()
        # Instead, just wait for the new game to load after clicking "Play Again"
        time.sleep(3)
        self.game_over_flag = None
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.actions.last_screenshot = None
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        self.match_over_detected = False
        return self._get_state()

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    @timing_decorator
    def step(self, action_index):
        # If match over, only allow no-op action (last action in list)
        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # No-op action

        self.current_cards = self.detect_cards_in_hand()
        self.logger.info(f"Current cards in hand: {self.current_cards}")

        if self.game_over_flag:
            done = True
            state = self._get_state()
            reward = self._compute_reward(state)
            result = self.game_over_flag
            if result == "victory":
                reward += 100
                self.logger.success("Victory detected - ending episode")
            elif result == "defeat":
                reward -= 100
                self.logger.success("Defeat detected - ending episode")

            return state, reward, done, result
        
        # If all cards are "Unknown", click at center and return no-op
        if all(card == "Unknown" for card in self.current_cards):
            self.logger.warning("All cards are Unknown, clicking at center and skipping move.")
            self.actions._click(640, 400)  # Click at center of screen in device coordinates
            # Return current state, zero reward, not done
            next_state = self._get_state()
            return next_state, 0, False, None

        action = self.available_actions[action_index]
        card_index, x_frac, y_frac = action
        self.logger.info(f"Action selected: card_index={card_index}, x_frac={x_frac:.2f}, y_frac={y_frac:.2f}")

        spell_penalty = 0
        state = self._get_state()

        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            self.logger.info(f"Attempting to play {card_name}")
            x = int(x_frac * self.actions.WIDTH)
            y = int(y_frac * self.actions.HEIGHT)
            self.actions.card_play(x, y, card_index)
            # time.sleep(1)  # You can reduce this if needed

            # --- Spell penalty logic ---
            if card_name in SPELL_CARDS:
                enemy_positions = []
                for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
                    ex = state[i]
                    ey = state[i + 1]
                    if ex != 0.0 or ey != 0.0:
                        ex_px = int(ex * self.actions.WIDTH)
                        ey_px = int(ey * self.actions.HEIGHT)
                        enemy_positions.append((ex_px, ey_px))
                radius = 100
                found_enemy = any((abs(ex - x) ** 2 + abs(ey - y) ** 2) ** 0.5 < radius for ex, ey in enemy_positions)
                self.logger.info(f"Spell used: {card_name}, found_enemy: {found_enemy}, enemy_positions: {enemy_positions}")
                if not found_enemy:
                    spell_penalty = -5  # Penalize for wasting spell

        # --- Princess tower reward logic ---
        current_enemy_princess_towers = self._count_enemy_princess_towers()
        princess_tower_reward = 0
        if self.prev_enemy_princess_towers is not None:
            if current_enemy_princess_towers < self.prev_enemy_princess_towers:
                princess_tower_reward = 20
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        done = False
        
        # Avoid recomputing state in order to save computation time
        reward = self._compute_reward(state) + spell_penalty + princess_tower_reward
        next_state = state
        return next_state, reward, done, None

    @timing_decorator
    def _get_state(self):
        # Get card info
        self.current_cards = self.detect_cards_in_hand()
        
        # Caching mechanism for screenshot, to speed up processing
        elixir = self.actions.count_elixir(self.actions.last_screenshot)

        card_info_flat = []
        for card_name in self.current_cards:
            card_id = self.card_to_id.get(card_name, self.card_to_id["unknown"])
            elixir_cost = self.card_data.get(card_name, {}).get("elixir", 5)
            if "elixir" not in self.card_data.get(card_name, {}):
                self.logger.warning(f"Card '{card_name}' detected but has no elixir linked to it, defaulting to 5.")
            card_info_flat.extend([card_id / len(self.card_to_id), elixir_cost / 10.0])

        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
        
        results = self.run_detection_workflow(workspace_name)

        # print("RAW results:", results)

        # Handle new structure: dict with "predictions" key
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]

        if not predictions:
            self.logger.warning("No predictions found in results")
            return None

        # After getting 'predictions' from results:
        if isinstance(predictions, dict) and "predictions" in predictions:
            predictions = predictions["predictions"]

        for p in predictions:
            self.logger.debug(f"{p['class']} at ({p['x']}, {p['y']}) with confidence {p['confidence']:.2f}")
        self.logger.debug(f"Detected classes: {[repr(p.get('class', '')) for p in predictions if isinstance(p, dict)]}")

        TOWER_CLASSES = {
            "ally king tower",
            "ally princess tower",
            "enemy king tower",
            "enemy princess tower"
        }

        def normalize_class(cls):
            return cls.strip().lower() if isinstance(cls, str) else ""

        allies = [
            (p["x"], p["y"])
            for p in predictions
            if (
                isinstance(p, dict)
                and normalize_class(p.get("class", "")) not in TOWER_CLASSES
                and normalize_class(p.get("class", "")).startswith("ally")
                and "x" in p and "y" in p
            )
        ]

        enemies = [
            (p["x"], p["y"])
            for p in predictions
            if (
                isinstance(p, dict)
                and normalize_class(p.get("class", "")) not in TOWER_CLASSES
                and normalize_class(p.get("class", "")).startswith("enemy")
                and "x" in p and "y" in p
            )
        ]
        
        
        self.logger.debug(f"Allies: {allies}")
        self.logger.debug(f"Enemies: {enemies}")

        # Normalize positions
        def normalize(units):
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x, y in units]

        # Pad or truncate to fixed length
        def pad_units(units, max_units):
            units = normalize(units)
            if len(units) < max_units:
                units += [(0.0, 0.0)] * (max_units - len(units))
            return units[:max_units]

        ally_positions = pad_units(allies, MAX_ALLIES)
        enemy_positions = pad_units(enemies, MAX_ENEMIES)

        # Flatten positions
        ally_flat = [coord for pos in ally_positions for coord in pos]
        enemy_flat = [coord for pos in enemy_positions for coord in pos]

        state = np.array([elixir / 10.0] + card_info_flat + ally_flat + enemy_flat, dtype=np.float32)
        return state

    @timing_decorator
    def run_detection_workflow(self, workspace_name):
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )
        
        return results

    @timing_decorator
    def _compute_reward(self, state):
        if state is None:
            return 0

        elixir = state[0] * 10

        # Sum all enemy positions (not just the first)
        enemy_positions = state[1 + 2 * MAX_ALLIES:]  # All enemy x1, y1, x2, y2, ...
        enemy_presence = sum(enemy_positions)

        reward = -enemy_presence

        # Elixir efficiency: reward for spending elixir if it reduces enemy presence
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            if elixir_spent > 0 and enemy_reduced > 0:
                reward += 2 * min(elixir_spent, enemy_reduced)  # tune this factor

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        return reward

    @timing_decorator
    def detect_cards_in_hand(self):
        try:
            card_paths = self.actions.capture_individual_cards()
            self.logger.debug("Testing individual card predictions:")

            cards = []
            workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            if not workspace_name:
                raise ValueError("WORKSPACE_CARD_DETECTION environment variable is not set. Please check your .env file.")
            
            for card_path in card_paths:
                results = self.run_card_detection_workflow(workspace_name, card_path)
                # print("Card detection raw results:", results)  # Debug print

                # Fix: parse nested structure
                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])
                if predictions:
                    card_name = predictions[0]["class"]
                    self.logger.debug(f"Detected card: {card_name}")
                    cards.append(card_name)
                else:
                    self.logger.debug("No card detected.")
                    cards.append("Unknown")
            return cards
        except Exception as e:
            self.logger.error(f"Error in detect_cards_in_hand: {e}")
            return []
        
    @timing_decorator
    def run_card_detection_workflow(self, workspace_name, card_path):
        results = self.card_model.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id="custom-workflow",
                    images={"image": card_path}
                )
        
        return results

    def get_available_actions(self):
        """Generate all possible actions"""
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # No-op action
        return actions

    def _endgame_watcher(self):
        """Thread that watches for both match over and game end conditions"""
        while not self._endgame_thread_stop.is_set():
            # Check for game end (victory/defeat screen)
            result = self.actions.detect_game_end()
            
            if result:
                self.logger.success(f"Game ended with result: {result}")
                self.game_over_flag = result
                time.sleep(5)
                
                self.actions.click_ok_button()
                break
            
            # Sleep a bit to avoid hammering the CPU
            time.sleep(0.3)

    def _count_enemy_princess_towers(self):
        # Considering count enemy princess is called only from step function and step function already has a **fresh** screenshot, we can skip taking a new screenshot here
        self.actions.capture_area(self.screenshot_path, self.actions.last_screenshot)
        
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
        
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]
        return sum(1 for p in predictions if isinstance(p, dict) and p.get("class") == "enemy princess tower")
