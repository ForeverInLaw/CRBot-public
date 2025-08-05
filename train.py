import os
import torch
import glob
import json
import csv
from env import ClashRoyaleEnv
from dqn_agent import DQNAgent
from pynput import keyboard
from datetime import datetime
import csv
from colors import bcolors

class KeyboardController:
    def __init__(self):
        self.should_exit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print(f"\n{bcolors.WARNING}Shutdown requested - cleaning up...{bcolors.ENDC}")
                self.should_exit = True
        except AttributeError:
            pass  # Special key pressed
            
    def is_exit_requested(self):
        return self.should_exit

def get_latest_model_path(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    model_files.sort()  # Lexicographical sort works for timestamps
    return model_files[-1]

def train():
    # Create a unique timestamp for this training session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure models directory exists
    log_dir = os.path.join("models", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"log_{session_timestamp}.csv")
    
    # Write header to log file
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward", "Epsilon", "Result"])

    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)

    # Load latest model if available
    latest_model = get_latest_model_path("models")
    if latest_model:
        agent.load(os.path.basename(latest_model))
        # Load epsilon
        meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                agent.epsilon = meta.get("epsilon", 1.0)
            print(f"Epsilon loaded: {agent.epsilon}")

    controller = KeyboardController()
    episodes = 10000
    batch_size = 32

    for ep in range(episodes):
        if controller.is_exit_requested():
            print(f"{bcolors.WARNING}Training interrupted by user.{bcolors.ENDC}")
            break

        state = env.reset()
        print(f"{bcolors.OKCYAN}Episode {ep + 1} starting. Epsilon: {agent.epsilon:.3f}{bcolors.ENDC}")
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, result = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            total_reward += reward

        print(f"{bcolors.OKGREEN}Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}, Result = {result}{bcolors.ENDC}")

        # Log episode data
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, total_reward, agent.epsilon, result])

        if ep % 10 == 0:
            agent.update_target_model()
            # Save model and epsilon every 10 episodes
            model_path = os.path.join("models", f"model_{session_timestamp}.pth")
            torch.save(agent.model.state_dict(), model_path)
            with open(os.path.join("models", f"meta_{session_timestamp}.json"), "w") as f:
                json.dump({"epsilon": agent.epsilon}, f)
            print(f"{bcolors.OKGREEN}Model and epsilon saved for session {session_timestamp}{bcolors.ENDC}")

if __name__ == "__main__":
    train()