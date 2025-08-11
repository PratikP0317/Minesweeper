import math
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Environment
from play_minesweeper_pygame import MinesweeperGame


def map_values_to_channel_indices(board: np.ndarray) -> np.ndarray:
    """
    Map Minesweeper cell values to 12 categorical channel indices:
    -3 -> 0 (mine shown when over)
    -2 -> 1 (flag)
    -1 -> 2 (unrevealed)
    0..8 -> 3..11
    """
    mapping = {
        -3: 0,
        -2: 1,
        -1: 2,
        0: 3,
        1: 4,
        2: 5,
        3: 6,
        4: 7,
        5: 8,
        6: 9,
        7: 10,
        8: 11,
    }
    vectorized = np.vectorize(lambda v: mapping[int(v)] if int(v) in mapping else 2)
    return vectorized(board).astype(np.int64)


def one_hot_encode_board(board: np.ndarray) -> torch.Tensor:
    """
    Convert HxW numpy board to 12xHxW one-hot torch tensor (float32).
    """
    idx = map_values_to_channel_indices(board)  # HxW with 0..11
    h, w = idx.shape
    out = np.zeros((12, h, w), dtype=np.float32)
    for c in range(12):
        out[c] = (idx == c).astype(np.float32)
    return torch.from_numpy(out)


class CNNActorCritic(nn.Module):
    """
    CNN backbone with spatial policy head (1x1 conv) producing per-cell logits and a value head.
    Works for variable board sizes because it keeps spatial dims and flattens at the end.
    """

    def __init__(self, in_channels: int = 12, channels: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Policy: 1x1 to one scalar per spatial location
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)

        # Value: compress channels, global average pool, linear
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.value_linear = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 12, H, W]
        action_mask: [B, H, W] boolean mask where True means valid action. If provided,
                      invalid logits are set to a very negative number.
        Returns:
            logits_flat: [B, H*W]
            value: [B]
        """
        features = self.backbone(x)
        policy_map = self.policy_head(features)  # [B, 1, H, W]
        logits = policy_map.squeeze(1)  # [B, H, W]

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        b, h, w = logits.shape
        logits_flat = logits.view(b, h * w)

        vfeat = F.relu(self.value_conv(features))  # [B, 32, H, W]
        vpool = vfeat.mean(dim=(2, 3))  # [B, 32]
        value = self.value_linear(vpool).squeeze(-1)  # [B]

        return logits_flat, value

    def act(self, x: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_flat, value = self.forward(x, action_mask)
        dist = Categorical(logits=logits_flat)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_flat, values = self.forward(x, action_mask)
        dist = Categorical(logits=logits_flat)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return logprobs, entropy, values


@dataclass
class PPOConfig:
    difficulty: str = "beginner"
    total_steps: int = 200_000
    batch_size: int = 4096
    update_epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_path: Optional[str] = "ppo_minesweeper.pt"


def get_action_mask_from_board(board: np.ndarray) -> np.ndarray:
    """Valid actions: cells with value == -1 (unrevealed)."""
    return (board == -1)


def index_to_coord(index: int, rows: int, cols: int) -> Tuple[int, int]:
    return int(index // cols), int(index % cols)


def coord_to_index(row: int, col: int, rows: int, cols: int) -> int:
    return int(row * cols + col)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rewards, values, dones: [T]
    Returns (advantages, returns) both [T]
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_advantage = 0.0
    last_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        advantages[t] = last_advantage
        last_value = values[t]

    returns = advantages + values
    return advantages, returns


def make_env(difficulty: str) -> MinesweeperGame:
    env = MinesweeperGame(difficulty=difficulty, render=False)
    return env


def train_ppo(config: PPOConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = make_env(config.difficulty)
    rows, cols = env.rows, env.cols

    device = torch.device(config.device)
    model = CNNActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    obs, _ = env.reset()

    global_step = 0
    episode = 0
    start_time = time.time()

    def empty_storage(capacity: int):
        return {
            "obs": [],  # tensors [12,H,W]
            "masks": [],  # tensors [H,W] bool
            "actions": [],  
            "logprobs": [],  
            "rewards": [],  
            "dones": [],  
            "values": [],  
        }

    storage = empty_storage(config.batch_size)

    while global_step < config.total_steps:
        storage = empty_storage(config.batch_size)

        steps_this_batch = 0
        while steps_this_batch < config.batch_size:
            obs_tensor = one_hot_encode_board(obs).unsqueeze(0).to(device)  # [1,12,H,W]
            mask_np = get_action_mask_from_board(obs)  # HxW bool
            if not mask_np.any():
                done = True
            else:
                done = False

            mask_tensor = torch.from_numpy(mask_np).to(device).unsqueeze(0)  # [1,H,W]

            with torch.no_grad():
                action_idx_tensor, logprob_tensor, value_tensor = model.act(obs_tensor, mask_tensor)

            action_idx = int(action_idx_tensor.item())
            action_row, action_col = index_to_coord(action_idx, rows, cols)

            next_obs, reward, terminated, truncated, info = env.step((action_row, action_col))
            done = bool(terminated or truncated)

            # Save transition
            storage["obs"].append(obs_tensor.squeeze(0).cpu())
            storage["masks"].append(mask_tensor.squeeze(0).cpu())
            storage["actions"].append(action_idx)
            storage["logprobs"].append(float(logprob_tensor.item()))
            storage["rewards"].append(float(reward))
            storage["dones"].append(float(done))
            storage["values"].append(float(value_tensor.item()))

            steps_this_batch += 1
            global_step += 1

            obs = next_obs

            if done:
                episode += 1
                obs, _ = env.reset()

            if global_step >= config.total_steps:
                break

        obs_batch = torch.stack(storage["obs"]).to(device)  # [B,12,H,W]
        mask_batch = torch.stack(storage["masks"]).to(device)  # [B,H,W]
        actions_batch = torch.tensor(storage["actions"], dtype=torch.int64, device=device)  # [B]
        old_logprobs_batch = torch.tensor(storage["logprobs"], dtype=torch.float32, device=device)  # [B]
        rewards_batch = torch.tensor(storage["rewards"], dtype=torch.float32, device=device)  # [B]
        dones_batch = torch.tensor(storage["dones"], dtype=torch.float32, device=device)  # [B]
        values_batch = torch.tensor(storage["values"], dtype=torch.float32, device=device)  # [B]

        advantages, returns = compute_gae(
            rewards=rewards_batch,
            values=values_batch,
            dones=dones_batch,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = obs_batch.size(0)
        idxs = np.arange(batch_size)

        for epoch in range(config.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_masks = mask_batch[mb_idx]
                mb_actions = actions_batch[mb_idx]
                mb_old_logprobs = old_logprobs_batch[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values = values_batch[mb_idx]

                new_logprobs, entropy, new_values = model.evaluate_actions(
                    mb_obs, mb_actions, mb_masks
                )

                # Policy loss
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value_clipped = mb_values + (new_values - mb_values).clamp(-config.clip_coef, config.clip_coef)
                v_loss1 = (new_values - mb_returns).pow(2)
                v_loss2 = (value_clipped - mb_returns).pow(2)
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                # Entropy bonus
                entropy_loss = -config.ent_coef * entropy

                loss = pg_loss + config.vf_coef * v_loss + entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        if global_step % (config.batch_size * 2) == 0:
            elapsed = time.time() - start_time
            fps = int(global_step / max(1e-6, elapsed))
            avg_reward = rewards_batch.mean().item()
            print(f"steps={global_step} episodes={episode} avg_reward={avg_reward:.3f} fps={fps}")

        if config.save_path and (global_step // config.batch_size) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
            }, config.save_path)

    # Final save
    if config.save_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
        }, config.save_path)
        print(f"Saved model to {config.save_path}")


def load_model(path: str, device: Optional[str] = None) -> CNNActorCritic:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = CNNActorCritic().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    cfg = PPOConfig()
    print(f"Training PPO on Minesweeper difficulty={cfg.difficulty} device={cfg.device}")
    train_ppo(cfg)


