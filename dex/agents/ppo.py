
import numpy as np
import torch
import torch.nn as nn

from ..components.normalizer import Normalizer
from ..modules.policies import StochasticActor
from ..modules.subnetworks import MLP
from ..utils.general_utils import AttrDict
from .base import BaseAgent


class PPO(BaseAgent):
    """On-policy PPO agent for goal-conditioned DEX environments."""

    def __init__(self, env_params, sampler, agent_cfg):
        super().__init__()

        self.discount = agent_cfg.discount
        self.gae_lambda = agent_cfg.gae_lambda
        self.reward_scale = agent_cfg.reward_scale
        self.reward_func = sampler.reward_func
        self.device = torch.device(agent_cfg.device)

        self.n_rollout_episodes = agent_cfg.n_rollout_episodes
        self.ppo_epochs = agent_cfg.ppo_epochs
        self.minibatch_size = agent_cfg.minibatch_size
        self.clip_ratio = agent_cfg.clip_ratio
        self.vf_coef = agent_cfg.vf_coef
        self.ent_coef = agent_cfg.ent_coef
        self.max_grad_norm = agent_cfg.max_grad_norm
        self.target_kl = agent_cfg.target_kl

        self.clip_obs = agent_cfg.clip_obs
        self.norm_clip = agent_cfg.norm_clip
        self.norm_eps = agent_cfg.norm_eps

        self.dima = env_params["act"]
        self.dimo = env_params["obs"]
        self.dimg = env_params["goal"]
        self.goal_conditioned = env_params["goal_conditioned"]
        self.max_action = env_params["max_action"]
        self.goal_conditioned = env_params["goal_conditioned"]

        self.o_norm = Normalizer(
            size=self.dimo,
            default_clip_range=self.norm_clip,
            eps=self.norm_eps,
        )
        self.g_norm = Normalizer(
            size=self.dimg,
            default_clip_range=self.norm_clip,
            eps=self.norm_eps,
        )

        self.input_dim = self.dimo + (
            self.dimg if self.goal_conditioned else 0
        )

        self.actor = StochasticActor(
            self.input_dim,
            self.dima * 2,
            agent_cfg.hidden_dim,
        ).to(self.device)

        self.value = MLP(
            self.input_dim,
            1,
            agent_cfg.hidden_dim,
        ).to(self.device)
        print("Actor device:", next(self.actor.parameters()).device)
        print("Value device:", next(self.value.parameters()).device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=agent_cfg.critic_lr
        )

        # One entry per completed episode.
        self.rollouts = []
        self.last_metrics = None

        # Action-time metadata. Must be reset after each episode.
        self._pending_log_probs = []

    def _normalised_inputs(self, obs, goals=None):
        if self.goal_conditioned:
            obs_proc, goals_proc = self._preproc_og(obs, goals)

            obs_norm = self.o_norm.normalize(obs_proc)
            goals_norm = self.g_norm.normalize(goals_proc)

            inputs = np.concatenate(
                [obs_norm, goals_norm], axis=-1
            )
        else:
            obs_proc = np.clip(
                obs, -self.clip_obs, self.clip_obs
            )
            obs_norm = self.o_norm.normalize(obs_proc)
            inputs = obs_norm

        return self.to_torch(inputs)

    def get_action(self, state, noise=False):
        with torch.no_grad():
            if self.goal_conditioned:
                obs = state["observation"]
                goal = state["desired_goal"]
            else:
                obs = state
                goal = None

            inputs = self._normalised_inputs(obs, goal)
            dist = self.actor(inputs)

            if noise:
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                self._pending_log_probs.append(
                    float(log_prob.reshape(-1)[0].cpu().item())
                )
            else:
                action = dist.mean

        action_np = action.detach().cpu().numpy().reshape(-1)
        return self.max_action * action_np


    def store_rollout(self, episode_batch):
        """Store one fresh episode together with its action-time log-probs."""
        log_probs = np.asarray(
            self._pending_log_probs, dtype=np.float32
        )
        self._pending_log_probs = []

        actions = np.asarray(episode_batch.actions)
        n_envs, horizon = actions.shape[:2]

        # This implementation expects the serial DEX sampler (one episode
        # per call). Do not silently associate log-probs with wrong episodes.
        if n_envs != 1:
            raise ValueError(
                "PPO store_rollout currently expects one episode per "
                f"sampler call; received {n_envs} episodes."
            )

        if log_probs.shape[0] != horizon:
            raise ValueError(
                "Number of saved action log-probabilities does not match "
                f"episode length: {log_probs.shape[0]} vs {horizon}. "
                "Check sampler calls and reset logic."
            )

        obs = np.asarray(episode_batch.obs)
        ag = np.asarray(episode_batch.ag)
        goals = np.asarray(episode_batch.g)

        # Goal is constant within an episode. Append it for the final state.
        goals_full = np.concatenate(
            [goals, goals[:, -1:, :]], axis=1
        )

        # Freeze normalized inputs BEFORE the trainer updates the normalizer.
        with torch.no_grad():
            inputs = self._normalised_inputs(obs, goals_full)

        self.rollouts.append(
            AttrDict(
                episode=episode_batch,
                inputs=inputs.detach(),
                old_log_probs=self.to_torch(log_probs),
            )
        )

    
    def _process_rollouts(self):
        if not self.rollouts:
            raise RuntimeError("Cannot process an empty PPO rollout buffer.")

        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_rewards = []

        for item in self.rollouts:
            ep = item.episode

            obs = np.asarray(ep.obs)
            ag = np.asarray(ep.ag)
            goals = np.asarray(ep.g)
            actions = np.asarray(ep.actions)
            dones = np.asarray(ep.dones)

            n_envs, horizon = actions.shape[:2]
            if n_envs != 1:
                raise ValueError("Expected one episode per rollout entry.")

            inputs = item.inputs
            old_log_probs = item.old_log_probs.reshape(-1)

            if inputs.shape[1] != horizon + 1:
                raise ValueError(
                    f"Expected T+1 observations, got {inputs.shape[1]} "
                    f"for T={horizon}."
                )

            if old_log_probs.numel() != horizon:
                raise ValueError("Incorrect old log-probability count.")

            # Compute rewards for the actual transitions.
            if self.goal_conditioned:
                rewards = self.reward_func(
                    ag[:, 1:].reshape(horizon, -1),
                    goals.reshape(horizon, -1),
                    None,
                )
                rewards = np.asarray(
                    rewards, dtype=np.float32
                ).reshape(1, horizon)
            else:
                rewards = np.asarray(
                    ep.rewards, dtype=np.float32
                ).reshape(1, horizon)

            rewards = rewards * self.reward_scale
            rewards_t = self.to_torch(rewards)

            # True means the environment reached a terminal state.
            # Do not overwrite the final done flag: genuine terminal
            # states must not bootstrap from their next-state value.
            done_mask = self.to_torch(
                dones.reshape(1, horizon).astype(np.float32)
            )
            nonterminal = 1.0 - done_mask

            with torch.no_grad():
                values = self.value(inputs).squeeze(-1)

            # GAE for this episode only; its length may vary.
            advantages = torch.zeros_like(rewards_t)
            last_gae = torch.zeros(
                (), device=self.device, dtype=torch.float32
            )

            for t in reversed(range(horizon)):
                delta = (
                    rewards_t[0, t]
                    + self.discount
                    * values[0, t + 1]
                    * nonterminal[0, t]
                    - values[0, t]
                )

                last_gae = (
                    delta
                    + self.discount
                    * self.gae_lambda
                    * nonterminal[0, t]
                    * last_gae
                )
                advantages[0, t] = last_gae

            returns = advantages + values[:, :-1]

            # Flatten transitions, not whole variable-length episodes.
            all_obs.append(
                inputs[:, :-1].reshape(-1, inputs.shape[-1])
            )
            all_actions.append(
                self.to_torch(actions).reshape(
                    -1, actions.shape[-1]
                ) / self.max_action
            )
            all_old_log_probs.append(old_log_probs)
            all_advantages.append(advantages.reshape(-1))
            all_returns.append(returns.reshape(-1))
            all_rewards.append(rewards_t.reshape(-1))

        batch = AttrDict(
            obs=torch.cat(all_obs, dim=0),
            actions=torch.cat(all_actions, dim=0),
            log_probs=torch.cat(all_old_log_probs, dim=0),
            adv=torch.cat(all_advantages, dim=0),
            returns=torch.cat(all_returns, dim=0),
        )

        mean_reward = float(
            torch.cat(all_rewards).mean().cpu().item()
        )

        return batch, mean_reward


    def update(self, replay_buffer=None, demo_buffer=None):
        # PPO only updates after collecting a complete batch of fresh episodes.
        if len(self.rollouts) < self.n_rollout_episodes:
            return self.last_metrics

        batch, mean_reward = self._process_rollouts()

        # This data must not be reused in another PPO update.
        self.rollouts = []

        adv = batch.adv
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n_samples = batch.obs.shape[0]
        n_updates = 0
        early_stop = False

        stats = {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(n_samples, device=batch.obs.device)

            for start in range(0, n_samples, self.minibatch_size):
                idx = permutation[start : start + self.minibatch_size]

                obs = batch.obs[idx]
                actions = batch.actions[idx]
                old_log_probs = batch.log_probs[idx]
                minibatch_adv = adv[idx]
                returns = batch.returns[idx]

                dist = self.actor(obs)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)

                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()

                surrogate1 = ratio * minibatch_adv
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * minibatch_adv

                entropy_actions = dist.rsample()
                entropy = -dist.log_prob(entropy_actions).sum(dim=-1).mean()
                actor_loss = (
                    -torch.minimum(surrogate1, surrogate2).mean()
                    - self.ent_coef * entropy
                )

                values = self.value(obs).squeeze(-1)
                value_loss = (values - returns).pow(2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                self.value_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.value.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean().item()
                    clip_frac = (
                        (ratio - 1.0).abs() > self.clip_ratio
                    ).float().mean().item()

                stats["actor_loss"] += actor_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                n_updates += 1

                if (
                    self.target_kl is not None
                    and approx_kl > 1.5 * self.target_kl
                ):
                    early_stop = True
                    break

            if early_stop:
                break

        if n_updates == 0:
            return self.last_metrics

        metrics = AttrDict(
            {key: value / n_updates for key, value in stats.items()}
        )
        metrics.update(
            AttrDict(
                value=batch.returns.mean().item(),
                batch_reward=mean_reward,
                n_updates=n_updates,
            )
        )

        self.last_metrics = metrics
        return metrics
