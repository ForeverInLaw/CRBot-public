import os
from constants import MODEL_DIR
import torch
import glob
import json
from core.environment import ClashRoyaleEnv
from core.dqn_agent import DQNAgent
from datetime import datetime
from utils.logger import Logger
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def get_latest_model_path():
    # Prefer new-style checkpoints
    ckpt_files = glob.glob(os.path.join(MODEL_DIR, "ckpt_*.pt"))
    if ckpt_files:
        ckpt_files.sort()
        return ckpt_files[-1]
    # Fallback to legacy weights
    model_files = glob.glob(os.path.join(MODEL_DIR, "model_*.pth"))
    if model_files:
        model_files.sort()
        return model_files[-1]
    return None

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional determinism vs speed trade-offs can be configured here

def train():
    logger = Logger(name="train")

    # Seeding
    seed = int(os.getenv("SEED", "42"))
    set_global_seeds(seed)
    logger.info(f"Using SEED={seed}")

    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)

    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    logs_dir = os.path.join(MODEL_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logs_dir)

    # Resume only if explicitly requested
    resume = os.getenv("RESUME_TRAINING", "1") == "1"
    if resume:
        latest_model = get_latest_model_path()
        if latest_model:
            print(f"Loading latest checkpoint: {latest_model}, {os.path.basename(latest_model)}")
            agent.load(os.path.basename(latest_model))
            logger.info(f"Resumed with epsilon: {agent.epsilon}")
        else:
            logger.warning("No checkpoint found to resume. Starting from scratch.")

    episodes = 10000
    batch_size = 32
    log_every = int(os.getenv("LOG_EVERY_STEPS", "20"))

    global_step = 0

    for ep in range(episodes):
        state = env.reset()
        logger.info(f"Episode {ep + 1} starting. Epsilon: {agent.epsilon:.3f}")
        total_reward = 0
        done = False
        step_i = 0
        while not done:
            step_i += 1
            global_step += 1
            # If state is None (no predictions), take NO_OP and continue
            if state is None:
                logger.warning("State is None; taking NO_OP and continuing.")
                action = env.action_size - 1  # NO_OP is last action
                next_state, reward, done = env.step(action)
                agent.after_step()
                state = next_state
                total_reward += reward
                # Log epsilon occasionally
                if log_every > 0 and (step_i % log_every == 0):
                    writer.add_scalar("train/epsilon", agent.epsilon, global_step)
                continue

            action_mask = env.get_valid_action_mask(state)
            action = agent.act(state, action_mask)
            next_state, reward, done = env.step(action)

            if next_state is None:
                logger.warning("Next state is None; skipping memory and replay for this step.")
                agent.after_step()
                state = next_state
                total_reward += reward
                if log_every > 0 and (step_i % log_every == 0):
                    writer.add_scalar("train/epsilon", agent.epsilon, global_step)
                continue

            next_action_mask = env.get_valid_action_mask(next_state)
            agent.remember(state, action, reward, next_state, done, next_action_mask)
            loss = agent.replay(batch_size)

            # After each env step, update schedules
            agent.after_step()

            state = next_state
            total_reward += reward

            # TensorBoard logging
            if loss is not None:
                writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/epsilon", agent.epsilon, global_step)
            if log_every > 0 and (step_i % log_every == 0):
                logger.info(f"t={step_i} a={action} r={reward:.2f} R={total_reward:.2f} eps={agent.epsilon:.3f}")

        # Episode summary
        writer.add_scalar("train/episode_reward", total_reward, ep + 1)
        logger.success(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

        if ep % 10 == 0:
            agent.update_target_model()
            # Save full checkpoint every 10 episodes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(MODEL_DIR, f"ckpt_{timestamp}.pt")
            agent.save(ckpt_path)
            logger.success(f"Checkpoint saved to {ckpt_path}")

    writer.close()

if __name__ == "__main__":
    train()