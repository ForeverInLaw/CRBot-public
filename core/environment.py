from constants import CURRENT_SCREENSHOT
import numpy as np
import time
import os
import threading
from dotenv import load_dotenv
from core.actions import Actions
from core.cards import CARDS_DATA, ID_TO_CARD, SPELL_CARDS
from inference_sdk import InferenceHTTPClient
from utils.decorator import timing_decorator
from utils.logger import Logger

# Load environment variables from .env file
load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10
NUM_CARDS = 4

class ClashRoyaleEnv:
    def __init__(self):
        self.logger = Logger(name="ClashRoyaleEnv")
        self.actions = Actions()
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        
        # Determine number of card types (by ID space) for one-hot encoding
        self.num_card_types = (max(ID_TO_CARD.keys()) + 1) if ID_TO_CARD else 1
        
        # State size: elixir (1) + ally/enemy positions + one-hot per card
        self.state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES) + NUM_CARDS * self.num_card_types

        self.grid_width = 18
        self.grid_height = 28

        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None

        self.prev_enemy_princess_towers = None

        self.match_over_detected = False
        
        # Keep last known valid state to reuse after match over
        self._last_state = None

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
        time.sleep(3)
        self.game_over_flag = None
        self.match_over_detected = False
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        # Ensure initial state includes detected cards
        self.current_cards = self.detect_cards_in_hand()
        state = self._get_state()
        self._last_state = state
        return state

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    @timing_decorator
    def step(self, action_index):
        # If match over, wait for final result while restricting actions to NO_OP
        if self.match_over_detected:
            state = self._last_state
            if state is None:
                neutral = np.zeros(self.state_size, dtype=np.float32)
                neutral[0] = 0.0
                state = neutral
            # If final result already known, return terminal reward
            if self.game_over_flag:
                result = self.game_over_flag
                reward = self._compute_reward(state)
                if result == "victory":
                    reward += 100
                elif result == "defeat":
                    reward -= 100
                return state, reward, True
            # Otherwise keep episode running with NO_OP only and zero reward (throttle loop)
            time.sleep(0.5)
            return state, 0.0, False

        # Capture pre-action state if needed for reward components
        pre_action_state = self._get_state()
        
        if self.game_over_flag:
            done = True
            reward = self._compute_reward(pre_action_state)
            result = self.game_over_flag
            if result == "victory":
                reward += 100
                self.logger.success("Victory detected - ending episode")
            elif result == "defeat":
                reward -= 100
                self.logger.warning("Defeat detected - ending episode")
            return pre_action_state, reward, done

        # Update current cards before choosing/playing
        self.current_cards = self.detect_cards_in_hand()
        self.logger.info(f"Current cards in hand: {self.current_cards}")

        # If all cards are "Unknown", click at center and return no-op
        if all(card == "Unknown" for card in self.current_cards):
            self.logger.warning("All cards are Unknown, clicking at center and skipping move.")
            self.actions._click(640, 400)
            # Rebuild state after click
            next_state = self._get_state()
            self._last_state = next_state
            return next_state, 0, False

        action = self.available_actions[action_index]
        self.logger.info(f"Action selected: {action}")

        spell_penalty = 0

        if action["type"] == "PLAY":
            card_index = action["card"]
            x_frac = action["x"]
            y_frac = action["y"]
            if card_index < len(self.current_cards):
                card_name = self.current_cards[card_index]
                self.logger.info(f"Attempting to play {card_name}")
                x = int(x_frac * self.actions.WIDTH)
                y = int(y_frac * self.actions.HEIGHT)
                self.actions.card_play(x, y, card_index)
                # --- Spell penalty logic ---
                if card_name.lower() in SPELL_CARDS and pre_action_state is not None:
                    self.logger.extra_visibility(f"Played spell card: {card_name}")
                    enemy_positions = []
                    for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
                        ex = pre_action_state[i]
                        ey = pre_action_state[i + 1]
                        if ex != 0.0 or ey != 0.0:
                            ex_px = int(ex * self.actions.WIDTH)
                            ey_px = int(ey * self.actions.HEIGHT)
                            enemy_positions.append((ex_px, ey_px))
                    radius = 100
                    radius2 = float(radius * radius)
                    found_enemy = any(((float(ex - x) * float(ex - x) + float(ey - y) * float(ey - y)) < radius2) for ex, ey in enemy_positions)
                    self.logger.extra_visibility(f"Spell hit enemy: {found_enemy} at positions {enemy_positions}")
                    if not found_enemy:
                        self.logger.warning(f"Spell {card_name} did not hit any enemy, applying penalty.")
                        spell_penalty = -5
        
        # Always rebuild next_state after action (including NO_OP)
        next_state = self._get_state()
        self._last_state = next_state
        
        # --- Princess tower reward logic ---
        current_enemy_princess_towers = self._count_enemy_princess_towers()
        princess_tower_reward = 0
        if self.prev_enemy_princess_towers is not None:
            if current_enemy_princess_towers < self.prev_enemy_princess_towers:
                self.logger.success("Enemy princess tower destroyed, applying reward.")
                princess_tower_reward = 20
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        done = False
        reward = self._compute_reward(next_state) + spell_penalty + princess_tower_reward
        return next_state, reward, done

    @timing_decorator
    def _get_state(self):
        screenshot = self.actions.capture_area(CURRENT_SCREENSHOT)
        elixir = self.actions.count_elixir(screenshot)
        
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
        
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": CURRENT_SCREENSHOT}
        )

        # Handle new structure: dict with "predictions" key
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]

        if not predictions:
            self.logger.warning("WARNING: No predictions found in results")
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

        # One-hot encode current cards into the state
        card_info = []
        for card_name in self.current_cards:
            card_data = CARDS_DATA.get(card_name.lower(), CARDS_DATA["unknown"]) if CARDS_DATA else {"id": 0}
            card_id = int(card_data.get("id", 0))
            one_hot = [0.0] * self.num_card_types
            if 0 <= card_id < self.num_card_types:
                one_hot[card_id] = 1.0
            card_info.extend(one_hot)
        
        # Pad with one-hot(unknown) if less than NUM_CARDS
        while len(card_info) < NUM_CARDS * self.num_card_types:
            unknown_hot = [0.0] * self.num_card_types
            unknown_hot[0] = 1.0  # unknown id = 0
            card_info.extend(unknown_hot)

        state = np.array([elixir / 10.0] + ally_flat + enemy_flat + card_info, dtype=np.float32)
        self._last_state = state
        return state

    def _compute_reward(self, state):
        if state is None:
            return 0

        elixir = float(state[0] * 10)

        # Extract enemy positions properly (excluding tower positions)
        enemy_positions = []
        for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
            ex = float(state[i])
            ey = float(state[i + 1])
            if ex != 0.0 or ey != 0.0:
                enemy_positions.append((ex, ey))
        
        enemy_presence = float(sum(abs(x) + abs(y) for x, y in enemy_positions)) if enemy_positions else 0.0
        
        self.logger.extra_visibility(f"Enemy valid positions count: {len(enemy_positions)}")
        
        reward = -enemy_presence * 0.5
        
        if self.prev_elixir is not None and elixir >= 8:
            reward -= 2.0
            self.logger.extra_visibility(f"Applied inaction penalty for high elixir: {elixir}")

        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = max(0.0, float(self.prev_elixir) - float(elixir))
            enemy_reduced = max(0.0, float(self.prev_enemy_presence) - float(enemy_presence))
            if elixir_spent > 0.0:
                reward += elixir_spent * 1.0
                if enemy_reduced > 0.0:
                    reward += 3.0 * min(elixir_spent, enemy_reduced)

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        self.logger.extra_visibility(f"Elixir: {elixir}, Enemy Presence: {enemy_presence}, Reward: {reward}")
        return reward

    @timing_decorator
    def detect_cards_in_hand(self):
        try:
            if self.match_over_detected:
                return self.current_cards if self.current_cards else ["Unknown"] * NUM_CARDS
            card_paths = self.actions.capture_individual_cards()
            self.logger.debug("\nTesting individual card predictions:")

            cards = []
            workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            if not workspace_name:
                raise ValueError("WORKSPACE_CARD_DETECTION environment variable is not set. Please check your .env file.")
            
            for card_path in card_paths:
                results = self.card_model.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id="custom-workflow",
                    images={"image": card_path}
                )
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
                    self.logger.warning("No card detected.")
                    cards.append("Unknown")
            return cards
        except Exception as e:
            self.logger.error(f"Error in detect_cards_in_hand: {e}")
            return []

    def get_available_actions(self):
        """Generate all possible actions"""
        actions = []
        for card_index in range(NUM_CARDS):
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    actions.append({"type": "PLAY", "card": card_index, "x": x / (self.grid_width - 1), "y": y / (self.grid_height - 1)})
        actions.append({"type": "NO_OP"})
        return actions

    def get_valid_action_mask(self, state):
        """Generate a mask of valid actions based on the current state."""
        mask = np.zeros(self.action_size, dtype=bool)
        # After match over, only NO_OP is valid
        if self.match_over_detected:
            mask[-1] = True
            return mask

        current_elixir = state[0] * 10

        # Decode one-hot card ids from the state
        base = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)
        card_ids = []
        for card_slot in range(NUM_CARDS):
            start = base + card_slot * self.num_card_types
            end = start + self.num_card_types
            if end > len(state):
                break
            segment = state[start:end]
            if len(segment) == self.num_card_types:
                card_id = int(np.argmax(segment)) if np.any(segment) else 0
                card_ids.append(card_id)

        # Determine if there are any enemies on the field (from state)
        any_enemy_present = False
        for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
            ex = state[i]
            ey = state[i + 1]
            if ex != 0.0 or ey != 0.0:
                any_enemy_present = True
                break

        for i, action in enumerate(self.available_actions):
            if action["type"] == "NO_OP":
                mask[i] = True
                continue

            card_index = action["card"]
            if card_index < len(card_ids):
                card_id = card_ids[card_index]
                if card_id != 0:  # Card exists
                    card_data = ID_TO_CARD.get(card_id)
                    if not card_data:
                        continue
                    # Do not allow spell cards if no enemies detected
                    card_name = card_data.get("name", "").lower()
                    if (card_name in SPELL_CARDS) and (not any_enemy_present):
                        continue
                    if current_elixir >= card_data["elixir"]:
                        mask[i] = True
        
        # Brief diagnostic: log counts of valid actions occasionally
        valid_count = int(mask.sum())
        if valid_count <= 1:  # likely only NO_OP
            self.logger.debug(f"Valid actions: {valid_count} (elixir={current_elixir}, card_ids={card_ids})")
        return mask

    def _endgame_watcher(self):
        """Thread that watches for both match over and game end conditions"""
        while not self._endgame_thread_stop.is_set():
            # Check for match over first (during game)
            if not self.match_over_detected and hasattr(self.actions, "detect_match_over"):
                if self.actions.detect_match_over():
                    self.logger.info("Match over detected (matchover.png), forcing no-op until next game.")
                    self.match_over_detected = True
            
            # Check for game end (victory/defeat screen)
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            
            time.sleep(0.5)

    def _count_enemy_princess_towers(self):
        self.actions.capture_area(CURRENT_SCREENSHOT)
        
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
        
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": CURRENT_SCREENSHOT}
        )
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]
        return sum(1 for p in predictions if isinstance(p, dict) and p.get("class") == "enemy princess tower")