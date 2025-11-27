# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any
from brax.training.acme import running_statistics
from brax.training import types
from brax.training.agents.clip_ed_sac import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition


# 추가: 선형 스케줄링 헬퍼 함수
def linear_schedule(current_step, start_value, end_value, total_steps):
    fraction = jnp.clip(current_step.lo.astype(jnp.float32) / total_steps, 0.0, 1.0)
    return start_value - fraction * (start_value - end_value)
    
def _compute_phi(q_params, normalizer_params, observations, actions):

    obs_norm = running_statistics.normalize(observations, normalizer_params)
    h = jnp.concatenate([obs_norm, actions], axis=-1)

    # ---- 자동 탐색 시작 ----
    root = q_params["params"]

    # QModule_* 찾기
    qmodule_key = [k for k in root.keys() if k.startswith("QModule")][0]
    qmodule = root[qmodule_key]

    # MLP_* 찾기
    mlp_key = [k for k in qmodule.keys() if k.startswith("MLP")][0]
    mlp = qmodule[mlp_key]

    # hidden_0 or Dense_0 찾기
    h0_key = [k for k in mlp.keys() if k.startswith("hidden") or k.startswith("Dense")][0]
    h0 = mlp[h0_key]

    w = h0["kernel"]
    b = h0["bias"]
    # ---- 자동 탐색 끝 ----

    phi = h @ w + b
    return phi
    
def make_losses(
        sac_network: sac_networks.SACNetworks,
        reward_scaling: float,
        discounting: float,
        action_size: int,
        start_beta: float = 0.5,
        end_beta: float = 0.0,
        anneal_beta: float = 1e5,
        start_clip: float = -50,
        end_clip: float = 50,
        anneal_clip: float = 1e5,
        auto_clip: bool = True,
        tau_q_range: float = 0.01
):
    """Creates the SAC losses."""

    target_entropy = -0.5 * action_size
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def alpha_loss(
            log_alpha: jnp.ndarray,
            policy_params: Params,
            normalizer_params: Any,
            transitions: Transition,
            key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def critic_loss(
            q_params: Params,
            policy_params: Params,
            normalizer_params: Any,
            target_q_params: Params,
            alpha: jnp.ndarray,
            transitions: Transition,
            key: PRNGKey,
            gradient_steps: jnp.ndarray,
            q_min: jnp.ndarray,
            q_max: jnp.ndarray,
    ) -> jnp.ndarray:
        #==========================================================        
        # Q(s,a) 계산 (기존 방식)
        q_old_action = q_network.apply(
            normalizer_params,
            q_params,
            transitions.observation,
            transitions.action,
        )

        # φ(s,a) = 첫 critic 첫 hidden layer pre-activation
        phi = _compute_phi(
            q_params,
            normalizer_params,
            transitions.observation,
            transitions.action,
        )
        #==========================================================
        # --- ---
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)

        #==========================================================        
        # target Q(s', a') 계산
        next_q = q_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )

        # φ(s',a') 계산
        phi_next = _compute_phi(
            target_q_params,
            normalizer_params,
            transitions.next_observation,
            next_action,
        )
        #========================================================== 
        term = discounting * phi_next - phi    # (batch, d)
        outer = jnp.einsum('bi,bj->bij', phi, term)  
        A_batch = jnp.mean(outer, axis=0)
        #==========================================================
        batch_min = jnp.min(q_old_action)
        batch_max = jnp.max(q_old_action)
        soft_update_min = batch_min
        soft_update_max = (1.0 - tau_q_range) * q_max + tau_q_range * batch_max
        hard_update_min = jnp.minimum(q_min, batch_min)
        hard_update_max = jnp.maximum(q_max, batch_max)
        is_initial_step = jnp.isinf(q_min)
        new_q_min = jax.lax.stop_gradient(
            jnp.where(is_initial_step, hard_update_min, soft_update_min)
        )
        new_q_max = jax.lax.stop_gradient(
            jnp.where(is_initial_step, hard_update_max, soft_update_max)
        )
        # next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob

        # --- Offline penalty start ---
        uncertainty = jnp.abs(next_q[..., 0] - next_q[..., 1])
        beta = linear_schedule(gradient_steps,
                               start_value = start_beta,
                               end_value = end_beta,
                               total_steps = anneal_beta,)
        min_next_q = jnp.min(next_q, axis=-1)
        next_v = min_next_q - beta * uncertainty - alpha * next_log_prob
        # --- Offline penalty end ---

        # --- Hard Clipping ---
        is_primacy_phase = (gradient_steps.lo.astype(jnp.float32) < anneal_clip)
        next_v_clipped = jnp.where(
            is_primacy_phase,
            jnp.clip(next_v, start_clip, end_clip),
            next_v
        )
        if auto_clip:
            next_v_clipped = jnp.clip(next_v_clipped, new_q_min, new_q_max)
        # --- ---

        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * next_v_clipped
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        truncation = transitions.extras['state_extras']['truncation']
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        #==========================================================
        return q_loss, {'new_q_min': new_q_min,
                        'new_q_max': new_q_max,
                        'phi': phi,
                        'phi_next': phi_next,
                        'A_batch': A_batch }
        #==========================================================
    def actor_loss(
            policy_params: Params,
            normalizer_params: Any,
            q_params: Params,
            alpha: jnp.ndarray,
            transitions: Transition,
            key: PRNGKey,
    ) -> jnp.ndarray:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q
        return jnp.mean(actor_loss)

    return alpha_loss, critic_loss, actor_loss
