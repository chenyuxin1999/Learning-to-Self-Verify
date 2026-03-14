#!/usr/bin/env bash
set -xeuo pipefail

ulimit -n 65535

export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_GPU_MONITOR=1
export RAY_DEBUG_POST_MORTEM=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

export RAY_worker_register_timeout_seconds=1800
export RAY_TASK_MAX_RETRIES=3
export RAY_memory=100000000000
export RAY_object_store_memory=50000000000
export RAY_NUM_CPUS=32


adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 10))
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

policy_loss_mode="vanilla"
loss_agg_mode="token-mean"

enable_filter_groups=False
train_prompt_bsz=128
train_prompt_mini_bsz=32
n_resp_per_prompt=8


# Algorithm
train_temperature=1.0
train_top_p=1.0
train_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.6
val_top_p=0.95
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False
gen_tp=1
sp_size=1

NUM_GPUS=2
NNODES=1


CKPTS_DIR=Your_ckpts_dir
TRAIN_FILE=Your_train_file
TEST_FILE=Your_test_file
MODEL_PATH=Your_model_path



# ray start --head --dashboard-host='0.0.0.0' --disable-usage-stats --num-cpus 32 --min-worker-port=10002 --max-worker-port=10501



ray job submit --verbose \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.truncation='left' \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.nccl_timeout=7200 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${train_temperature} \
    actor_rollout_ref.rollout.top_p=${train_top_p} \
    actor_rollout_ref.rollout.top_k="${train_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=80 \
    trainer.total_training_steps=1000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
