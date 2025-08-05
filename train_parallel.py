import os
import torch
import torch.multiprocessing as mp
import glob
import json
import csv
from env import ClashRoyaleEnv
from dqn_agent import DQNAgent, DQN
from datetime import datetime
import time
from dotenv import load_dotenv
from logger import Logger
import traceback
import random

# Load environment variables from .env file
load_dotenv()

def get_latest_model_path(models_dir="models"):
    """
    Gets the path of the latest saved model in the specified directory.
    Prioritizes parallel models.
    """
    parallel_model_files = glob.glob(os.path.join(models_dir, "model_parallel_*.pth"))
    if parallel_model_files:
        return os.path.basename(max(parallel_model_files, key=os.path.getmtime))
    
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    
    return os.path.basename(max(model_files, key=os.path.getmtime))

def worker(device_serial, experience_queue, model_update_queue, results_queue):
    """
    The Actor process that interacts with the game environment.
    It is a lightweight process that only holds the model for acting.
    """
    try:
        logger = Logger(name="Worker", device_serial=device_serial, log_level="INFO")
        logger.info(f"Starting worker for device: {device_serial}")
        
        env = ClashRoyaleEnv(device_serial=device_serial)
        # The worker only needs the model structure, not the full agent
        model = DQN(env.state_size, env.action_size)
        epsilon = 1.0 # Start with max exploration

        episodes = 0
        while True:
            # Check for a new model from the learner
            if not model_update_queue.empty():
                update_data = model_update_queue.get()
                model.load_state_dict(update_data['model_state_dict'])
                epsilon = update_data['epsilon']
                model.eval() # Set model to evaluation mode
                logger.info(f"Model updated. New epsilon: {epsilon:.3f}")

            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Act using the current model and epsilon
                if random.random() < epsilon:
                    action = random.randrange(env.action_size)
                else:
                    with torch.no_grad():
                        q_values = model(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax().item()

                next_state, reward, done, result = env.step(action)
                
                # Send experience to the learner
                experience_queue.put((state, action, reward, next_state, done))
                
                state = next_state
                total_reward += reward

            episodes += 1
            # Send final results to the learner for logging
            results_queue.put({'total_reward': total_reward, 'result': result})
            logger.success(f"Completed episode {episodes} with reward {total_reward:.2f}, result: {result}")

    except Exception as e:
        logger.error(f"Exception in worker: {traceback.format_exc()}")

def train_parallel():
    """
    The Learner process that trains the central model.
    """
    # --- Setup ---
    logger = Logger(name="Learner", log_level="INFO", log_to_file=True)
    
    device_serials_str = os.getenv("DEVICE_SERIALS")
    if not device_serials_str:
        logger.error("DEVICE_SERIALS not set in .env file. Please set it to a comma-separated list of device serials.")
        return
        
    device_serials = [s.strip() for s in device_serials_str.split(',')]
    num_workers = len(device_serials)
    logger.info(f"Starting parallel training with {num_workers} workers.")

    # Dummy env to get state/action sizes
    env = ClashRoyaleEnv(device_serial=device_serials[0])
    # The learner has the full agent
    agent = DQNAgent(env.state_size, env.action_size)
    
    # Load latest model if available
    latest_model_filename = get_latest_model_path("models")
    if latest_model_filename:
        logger.info(f"Found latest model: {latest_model_filename}")
        agent.load(latest_model_filename)
        # Load epsilon if meta file exists
        meta_path = latest_model_filename.replace("model_parallel_", "meta_parallel_").replace("model_", "meta_").replace(".pth", ".json")
        meta_full_path = os.path.join("models", meta_path)
        if os.path.exists(meta_full_path):
            with open(meta_full_path, "r") as f:
                meta = json.load(f)
                agent.epsilon = meta.get("epsilon", 1.0)
            logger.info(f"Epsilon loaded: {agent.epsilon}")

    # --- Logging Setup ---
    log_dir = "models/logs"
    os.makedirs(log_dir, exist_ok=True)
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f"training_log_parallel_{session_timestamp}.csv")
    
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward", "Epsilon", "Result"])

    # --- Multiprocessing Queues ---
    experience_queue = mp.Queue()
    results_queue = mp.Queue()
    model_update_queues = [mp.Queue() for _ in range(num_workers)]

    # --- Start Workers ---
    processes = []
    for i, serial in enumerate(device_serials):
        p = mp.Process(target=worker, args=(serial, experience_queue, model_update_queues[i], results_queue))
        p.start()
        processes.append(p)

    # --- Main Learning Loop ---
    batch_size = 32
    episodes_completed = 0
    last_save_time = time.time()
    SAVE_INTERVAL_SECONDS = 120 # Save every 2 minutes
    TARGET_UPDATE_INTERVAL_EPISODES = 10 # Update target network every 10 episodes

    try:
        # Initial model distribution
        logger.info("Distributing initial model to workers...")
        update_data = {
            'model_state_dict': agent.model.state_dict(),
            'epsilon': agent.epsilon
        }
        for q in model_update_queues:
            q.put(update_data)

        while episodes_completed < 10000:
            # Collect experiences and train
            if not experience_queue.empty():
                state, action, reward, next_state, done = experience_queue.get()
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            # Check for completed episodes to log
            if not results_queue.empty():
                result_data = results_queue.get()
                episodes_completed += 1
                
                with open(log_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episodes_completed, result_data['total_reward'], agent.epsilon, result_data['result']])
                logger.info(f"Episode {episodes_completed} completed. Total Reward: {result_data['total_reward']:.2f}, Result: {result_data['result']}, Epsilon: {agent.epsilon:.3f}")

                if episodes_completed % TARGET_UPDATE_INTERVAL_EPISODES == 0:
                    agent.update_target_model()
                    logger.info("Target model updated.")

            # Periodically save model and distribute to workers
            if time.time() - last_save_time > SAVE_INTERVAL_SECONDS:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join("models", f"model_parallel_{timestamp}.pth")
                meta_path = os.path.join("models", f"meta_parallel_{timestamp}.json")
                
                torch.save(agent.model.state_dict(), model_path)
                with open(meta_path, "w") as f:
                    json.dump({"epsilon": agent.epsilon}, f)

                # Share the new model with the workers
                update_data = {
                    'model_state_dict': agent.model.state_dict(),
                    'epsilon': agent.epsilon
                }
                for q in model_update_queues:
                    # Clear old updates before putting a new one
                    while not q.empty():
                        q.get()
                    q.put(update_data)
                
                logger.success(f"Model saved to {model_path}. Distributed to workers. {episodes_completed} episodes completed.")
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