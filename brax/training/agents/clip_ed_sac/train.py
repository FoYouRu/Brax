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

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.clip_ed_sac import checkpoint
from brax.training.agents.clip_ed_sac import losses as sac_losses
from brax.training.agents.clip_ed_sac import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: types.UInt64
    env_steps: types.UInt64
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    q_min: jnp.ndarray
    q_max: jnp.ndarray
    A: jnp.ndarray 

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
        key: PRNGKey,
        obs_size: int,
        local_devices_to_use: int,
        sac_network: sac_networks.SACNetworks,
        alpha_optimizer: optax.GradientTransformation,
        policy_optimizer: optax.GradientTransformation,
        q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype('float32'))
    )

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=types.UInt64(hi=0, lo=0),
        env_steps=types.UInt64(hi=0, lo=0),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
        q_min=jnp.asarray(jnp.inf, dtype=jnp.float32),
        q_max=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        A=jnp.zeros((256, 256), dtype=jnp.float32)
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


def train(
        environment: envs.Env,
        num_timesteps,
        episode_length: int,
        wrap_env: bool = True,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        action_repeat: int = 1,
        num_envs: int = 1,
        num_eval_envs: int = 128,
        learning_rate: float = 1e-4,
        discounting: float = 0.9,
        seed: int = 0,
        batch_size: int = 256,
        num_evals: int = 1,
        normalize_observations: bool = False,
        max_devices_per_host: Optional[int] = None,
        reward_scaling: float = 1.0,
        tau: float = 0.005,
        min_replay_size: int = 0,
        max_replay_size: Optional[int] = None,
        grad_updates_per_step: int = 1,
        deterministic_eval: bool = False,
        network_factory: types.NetworkFactory[
            sac_networks.SACNetworks
        ] = sac_networks.make_sac_networks,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        eval_env: Optional[envs.Env] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        checkpoint_logdir: Optional[str] = None,
        restore_checkpoint_path: Optional[str] = None,
        # 추가: 불확실성 페널티를 위한 하이퍼파라미터
        start_beta: float = 0.0,
        end_beta: float = 0.0,
        anneal_beta: float = 1e5,
        # Reset 관련 인자 추가 ---
        reset_frequency: Optional[int] = None,
        reset_actor: bool = True,
        reset_critic: bool = True,
        reset_optimizer: bool = True,
        # Clip 관련 인자 추가 ---
        start_clip: float = -50,
        end_clip: float = 50,
        anneal_clip: float = 1e5,
        # 
        auto_clip: bool = True,
        tau_q_range: float = 0.01
):
    """SAC training."""
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info(
        'local_device_count: %s; total_device_count: %s',
        local_devices_to_use,
        device_count,
    )

    if min_replay_size >= num_timesteps:
        raise ValueError(
            'No training will happen because min_replay_size >= num_timesteps'
        )

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
            -(num_timesteps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    if wrap_env:
        if wrap_env_fn is not None:
            wrap_for_training = wrap_env_fn
        elif isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            raise ValueError('Unsupported environment type: %s' % type(env))

        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(
                    key, num_envs // jax.process_count() // local_devices_to_use
                ),
            )
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )  # pytype: disable=wrong-keyword-args

    obs_size = env.observation_size
    if isinstance(obs_size, Dict):
        raise NotImplementedError('Dictionary observations not implemented in SAC')
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)

    # 1. Define the custom learning rate schedule function for alpha
    def alpha_lr_schedule_fn(step: Union[int, jnp.ndarray]) -> jnp.ndarray:
        """Returns 0.0 LR for steps < anneal_beta, then base_alpha_lr."""
        anneal_steps = int(anneal_beta)
        lr = jnp.where(step < anneal_steps, 3e-4, 3e-4)
        return jnp.astype(lr, jnp.float32)
    alpha_optimizer = optax.adam(learning_rate=alpha_lr_schedule_fn)

    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step // device_count,
    )

    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=action_size,
        # 추가: make_losses에 새로운 하이퍼파라미터 전달
        start_beta = start_beta,
        end_beta = end_beta,
        anneal_beta = anneal_beta,
        start_clip = start_clip,
        end_clip = end_clip,
        anneal_clip = anneal_clip,
        auto_clip = auto_clip,
        tau_q_range = tau_q_range,
    )
    alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )
    actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )

    def sgd_step(
            carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)
        (critic_loss, aux), q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            # additional variables
            training_state.gradient_steps,
            training_state.q_min,
            training_state.q_max,
            optimizer_state=training_state.q_optimizer_state,
        )
        new_q_min = aux['new_q_min']
        new_q_max = aux['new_q_max']
        A_batch = aux['A_batch']
        
        # A_prev = training_state.A
        # decay = 0.99
        # A_new = decay * A_prev + (1 - decay) * A_batch
        # eigs = jnp.linalg.eigvals(A_new)
        
        A_new = A_batch                    
        eigs = jnp.linalg.eigvals(A_batch)
        
        lambda_max = jnp.max(jnp.real(eigs))

        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.target_q_params,
            q_params,
        )

        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params),
            'q_clipping/q_min_observed': new_q_min,
            'q_clipping/q_max_observed': new_q_max,
            'lambda_max': lambda_max,
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
            q_min=new_q_min,
            q_max=new_q_max,
            A=A_new
        )
        return (new_training_state, key), metrics

    def get_experience(
            normalizer_params: running_statistics.RunningStatisticsState,
            policy_params: Params,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        policy = make_policy((normalizer_params, policy_params))
        env_state, transitions = acting.actor_step(
            env, env_state, policy, key, extra_fields=('truncation',)
        )

        normalizer_params = running_statistics.update(
            normalizer_params,
            transitions.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    def training_step(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
    ) -> Tuple[
        TrainingState,
        envs.State,
        ReplayBufferState,
        Metrics,
    ]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        buffer_state, transitions = replay_buffer.sample(buffer_state)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions
        )

        metrics['buffer_current_size'] = replay_buffer.size(
            buffer_state)  # pytype: disable=unsupported-operands  # lax-types
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(
        prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
    )

    def training_epoch(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
                      env_steps_per_actor_step * num_training_steps_per_epoch
              ) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
    )
    del global_key

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        training_state = training_state.replace(
            normalizer_params=params[0],
            policy_params=params[1],
        )

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(
        env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
    )
    env_state = jax.pmap(env.reset)(env_keys)

    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    if not eval_env:
        eval_env = environment
    if wrap_env:
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
            )
        eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )  # pytype: disable=wrong-keyword-args

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (training_state.normalizer_params, training_state.policy_params)
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )

    replay_size = (
            jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    )
    logging.info('replay size after prefill %s', replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    # --- [고칠 부분 시작]: 마지막 리셋 시점 추적 변수 추가 ---
    last_reset_grad_step = types.UInt64(hi=0, lo=0)
    # --- [고칠 부분 끝]: 마지막 리셋 시점 추적 변수 추가 ---
    for _ in range(num_evals_after_init):
        logging.info('step %s', current_step)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = (
            training_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
        )
        current_step = int(_unpmap(training_state.env_steps))

        # --- [고침]: 리셋 로직 ---
        current_grad_steps_uint64 = _unpmap(training_state.gradient_steps)
        current_grad_steps_int = int(current_grad_steps_uint64.lo)  # Convert UInt64 low part to int
        if (
                reset_frequency is not None
                and (current_grad_steps_uint64.lo - last_reset_grad_step.lo) >= reset_frequency
        ):
            logging.info(f'Resetting agent at {current_grad_steps_int} gradient steps...')
            last_reset_grad_step = current_grad_steps_uint64  # 다음 리셋 기준 업데이트

            # 1. 새 키 생성 (호스트에서)
            local_key, key_policy_reset, key_q_reset = jax.random.split(local_key, 3)

            # 2. 호스트에서 파라미터 및 옵티마이저 상태 재초기화
            host_state = _unpmap(training_state)

            new_policy_params = host_state.policy_params
            new_policy_opt_state = host_state.policy_optimizer_state
            if reset_actor:
                logging.info('Resetting actor parameters...')
                new_policy_params = sac_network.policy_network.init(key_policy_reset)
                if reset_optimizer:
                    logging.info('Resetting actor optimizer state...')
                    new_policy_opt_state = policy_optimizer.init(new_policy_params)

            new_q_params = host_state.q_params
            new_q_opt_state = host_state.q_optimizer_state
            new_target_q_params = host_state.target_q_params
            if reset_critic:
                logging.info('Resetting critic parameters...')
                new_q_params = sac_network.q_network.init(key_q_reset)
                new_target_q_params = new_q_params
                if reset_optimizer:
                    logging.info('Resetting critic optimizer state...')
                    new_q_opt_state = q_optimizer.init(new_q_params)

            # 3. 호스트에서 TrainingState 업데이트 (리셋된 값들로)
            host_state = host_state.replace(
                policy_params=new_policy_params,
                policy_optimizer_state=new_policy_opt_state,
                q_params=new_q_params,
                q_optimizer_state=new_q_opt_state,
                target_q_params=new_target_q_params,
            )

            # 4. 업데이트된 상태를 모든 장치에 다시 복제
            training_state = jax.device_put_replicated(
                host_state, jax.local_devices()[:local_devices_to_use]
            )
            pmap.assert_is_replicated(training_state)  # 복제 확인
            logging.info('Agent reset complete.')
        # --- [고칠 부분 끝]: 리셋 로직 ---

        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                params = _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                )
                ckpt_config = checkpoint.network_config(
                    observation_size=obs_size,
                    action_size=env.action_size,
                    normalize_observations=normalize_observations,
                    network_factory=network_factory,
                )
                checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)

            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f'Total steps {total_steps} is less than `num_timesteps`='
            f' {num_timesteps}.'
        )

    params = _unpmap(
        (training_state.normalizer_params, training_state.policy_params)
    )

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
