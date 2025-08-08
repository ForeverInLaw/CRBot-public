import numpy as np
from typing import Any, List, Tuple


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.pos = 0
        self.size = 0
        self.data: List[Any] = [None] * self.capacity
        self.priorities: np.ndarray = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority: float = 1.0
        self.eps_priority: float = 1e-6

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Any):
        idx = self.pos
        self.data[idx] = transition
        # Assign max priority so new samples are likely to be seen
        self.priorities[idx] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        assert self.size > 0, "Buffer is empty"
        batch_size = min(batch_size, self.size)
        prios = self.priorities[: self.size]
        # Avoid all-zero priorities
        if prios.max() <= 0:
            prios = np.ones_like(prios)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        samples = [self.data[i] for i in indices]

        # Importance-sampling weights
        N = self.size
        weights = (N * probs[indices]) ** (-beta)
        # Normalize by max weight to keep in [0,1]
        weights /= weights.max() if weights.max() > 0 else 1.0
        return samples, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.maximum(priorities, self.eps_priority)
        for idx, p in zip(indices, priorities):
            self.priorities[int(idx)] = float(p)
            if p > self.max_priority:
                self.max_priority = float(p) 