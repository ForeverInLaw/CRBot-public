import os
import torch
import torch.multiprocessing as mp
import glob
import json
import csv
from env import ClashRoyaleEnv
from dqn_agent import DQNAgent
from datetime import datetime
import time
from dotenv import load_dotenv
from logger import Logger
import traceback

# Load environment variables from .env file
load_dotenv()


def get_latest_model_path(models_dir="models"):
    """
    Gets the path of the latest saved model in the specified directory.
    """
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    # Sort files by modification time to get the latest one
    latest_file = max(model_files, key=os.path.getmtime)
    return os.path.basename(latest_file)

def worker(device_serial, experience_queue, model_path_queue):
    """
    The Actor process that interacts with the game environment.
    """
    try:
        logger = Logger(name="Worker", device_serial=device_serial, log_level="DEBUG")
        logger.info(f"Starting worker for device: {device_serial}")
        env = ClashRoyaleEnv(device_serial=device_serial)
        agent = DQNAgent(env.state_size, env.action_size)
        
        logger.info("Loading latest model")
        # Load latest model if available
        latest_model_filename = get_latest_model_path("models")
        if latest_model_filename:
            logger.info(f"Found latest model: {latest_model_filename}")
            agent.load(latest_model_filename)
        logger.info(f"Loaded model: {latest_model_filename}")
        
        episodes = 0
        while True:
            # Check for a new model from the learner
            if not model_path_queue.empty():
                latest_model = model_path_queue.get()
                if latest_model:
                    try:
                        agent.load(latest_model)
                    except Exception as e:
                        logger.error(f"Error loading model: {e}")

            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, result = env.step(action)
                
                experience_queue.put((state, action, reward, next_state, done, result))
                
                state = next_state
                total_reward += reward

            episodes += 1
            logger.success(f"Completed episode {episodes} with reward {total_reward:.2f}, result: {result}")

    except Exception as e:
        logger.error(f"Exception in worker: {traceback.format_exc()}")

def train_parallel():
    """
    The Learner process that trains the model.
    """
    # --- Setup ---
    logger = Logger(name="Learner", log_level="DEBUG", log_to_file=True)
    
    device_serials_str = os.getenv("DEVICE_SERIALS")
    if not device_serials_str:
        logger.error("DEVICE_SERIALS not set in .env file. Please set it to a comma-separated list of device serials.")
        return
        
    device_serials = [s.strip() for s in device_serials_str.split(',')]
    num_workers = len(device_serials)
    logger.info(f"Starting parallel training with {num_workers} workers.")

    # Dummy env to get state/action sizes
    env = ClashRoyaleEnv(device_serial=device_serials[0])
    agent = DQNAgent(env.state_size, env.action_size)
    
    # --- Logging Setup ---
    log_dir = "models/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_log_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward", "Epsilon", "Result"])

    # --- Multiprocessing Queues ---
    experience_queue = mp.Queue()
    model_path_queue = mp.Queue()

    # --- Start Workers ---
    processes = []
    for serial in device_serials:
        p = mp.Process(target=worker, args=(serial, experience_queue, model_path_queue))
        p.start()
        processes.append(p)

    # --- Main Learning Loop ---
    batch_size = 32
    episodes_completed = 0
    last_save_time = time.time()

    try:
        while episodes_completed < 10000:
            # Collect experiences and train
            if not experience_queue.empty():
                state, action, reward, next_state, done, result = experience_queue.get()
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

                if done:
                    episodes_completed += 1
                    with open(log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([episodes_completed, reward, agent.epsilon, result])
                    logger.info(f"Episode {episodes_completed} completed. Epsilon: {agent.epsilon:.3f}")

            # Periodically update target model and save
            if time.time() - last_save_time > 60: # Every minute
                agent.update_target_model()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join("models", f"model_parallel_{timestamp}.pth")
                torch.save(agent.model.state_dict(), model_path)
                
                # Share the new model with the workers
                for _ in range(num_workers):
                    model_path_queue.put(model_path)
                
                logger.success(f"Model saved to {model_path}. {episodes_completed} episodes completed.")
                last_save_time = time.time()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        # --- Cleanup ---
        for p in processes:
            p.terminate()
            p.join()
        logger.info("All worker processes terminated.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train_parallel()
