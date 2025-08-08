import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from constants import MODEL_DIR
from utils.decorator import timing_decorator

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.feature(x)
        advantages = self.advantage(features)
        values = self.value(features)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
        self.per_beta = 0.4
        self.per_beta_increment = 1e-4  # anneal beta towards 1.0
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997  # per-step decay
        self.action_size = action_size
        
        # Step-based scheduling
        self.steps_done = 0
        self.target_update_every_steps = 1000

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def after_step(self):
        self.steps_done += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.per_beta < 1.0:
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
        if self.steps_done % self.target_update_every_steps == 0:
            self.update_target_model()

    def remember(self, s, a, r, s2, done, next_action_mask=None):
        self.memory.add((s, a, r, s2, done, next_action_mask))

    def act(self, state, action_mask=None):
        if random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        
        if action_mask is not None:
            torch_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values[0][~torch_mask] = -1e9
            
        return q_values.argmax().item()

    @timing_decorator
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        # Sample with priorities
        samples, indices, is_weights = self.memory.sample(batch_size, beta=self.per_beta)
        states, actions, rewards, next_states, dones, next_masks = zip(*samples)

        # Tensors on device
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)
        is_w_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        # Build next action mask tensor (True = valid). If None, consider all valid
        if any(m is not None for m in next_masks):
            mask_list = []
            for m in next_masks:
                if m is None:
                    mask_list.append(np.ones(self.action_size, dtype=bool))
                else:
                    mask_list.append(np.asarray(m, dtype=bool))
            next_mask_t = torch.as_tensor(np.stack(mask_list, axis=0), dtype=torch.bool, device=self.device)
        else:
            next_mask_t = torch.ones((len(samples), self.action_size), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # Online selects
            q_next_online = self.model(next_states_t)  # [B, A]
            q_next_online = q_next_online.masked_fill(~next_mask_t, -1e9)
            next_actions_t = torch.argmax(q_next_online, dim=1)  # [B]
            # Target evaluates
            q_next_target_all = self.target_model(next_states_t)  # [B, A]
            next_q_t = q_next_target_all.gather(1, next_actions_t.unsqueeze(1)).squeeze(1)  # [B]
            target_t = rewards_t + self.gamma * next_q_t * (~dones_t).float()

        # Current Q(s,a)
        q_current_all = self.model(states_t)  # [B, A]
        q_sa = q_current_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # [B]

        # TD error and weighted loss
        td_error = target_t.detach() - q_sa
        loss_per_item = self.criterion(q_sa, target_t.detach())
        loss = (is_w_t * loss_per_item).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities with absolute TD error
        new_priorities = (td_error.abs().detach().cpu().numpy() + 1e-6)
        self.memory.update_priorities(indices, new_priorities)

    def save(self, filename: str):
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join(MODEL_DIR, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "target_update_every_steps": self.target_update_every_steps,
            "action_size": self.action_size,
            "dueling": True,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load(self, filename):
        # Look in models/ directory by default
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join(MODEL_DIR, filename)
        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu")
        current_state = self.model.state_dict()
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New-style checkpoint
            model_sd = checkpoint.get("model_state_dict", {})
            filtered = {}
            skipped = []
            for k, v in model_sd.items():
                if k in current_state and current_state[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            self.model.load_state_dict(filtered, strict=False)
            # Restore optimizer if shapes match partially (safe to load strict=False)
            opt_sd = checkpoint.get("optimizer_state_dict")
            if opt_sd is not None:
                try:
                    self.optimizer.load_state_dict(opt_sd)
                except Exception:
                    print("Warning: optimizer state not loaded due to mismatch.")
            # Restore meta
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.steps_done = checkpoint.get("steps_done", self.steps_done)
            self.target_update_every_steps = checkpoint.get("target_update_every_steps", self.target_update_every_steps)
            loaded_count = len(filtered)
            total_count = len(current_state)
            print(f"Loaded checkpoint {path} (matched: {loaded_count}/{total_count}, skipped: {len(skipped)})")
            if skipped:
                print(f"Skipped keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        else:
            # Legacy raw state_dict of model weights
            filtered = {}
            skipped = []
            for k, v in checkpoint.items():
                if k in current_state and current_state[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            self.model.load_state_dict(filtered, strict=False)
            print(f"Loaded legacy weights {path} (matched: {len(filtered)}/{len(current_state)}, skipped: {len(skipped)})")
            if skipped:
                print(f"Skipped keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        self.model.to(self.device).eval()
        self.update_target_model()