# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Self-Verification DAPO Trainer with Ray-based single controller.
This trainer implements self-verification after standard DAPO training.
"""

import os
import random
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

from verl.utils.reward_score.math_dapo import normalize_final_answer, last_boxed_only_string, remove_boxed
import re



class RayDAPOSelfVerificationTrainer(RayPPOTrainer):
    """
    Self-Verification DAPO Trainer that implements self-verification after standard DAPO training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verification_buffer = []
        self.target_size = 0

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def compute_kl_related_metrics_for_verification(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Compute KL related metrics for verification task with verification_ prefix."""
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob_verification", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"verification_actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref_verification", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _extract_response_text(self, batch: DataProto, idx: int, prompt_ids: torch.Tensor) -> str:
        """Extract valid response text from batch at given index."""
        response_ids = batch.batch["responses"][idx]
        valid_response_length = batch.batch["attention_mask"][idx][len(prompt_ids):].sum()
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _extract_answer_text(self, batch: DataProto, idx: int, prompt_ids: torch.Tensor) -> str:
        """Extract answer text from batch at given index."""
        responses = self._extract_response_text(batch, idx, prompt_ids)
        answer_text = responses[-300:]

        boxed_pred = last_boxed_only_string(answer_text)
        extracted_pred = remove_boxed(boxed_pred) 
        
        pred = normalize_final_answer(extracted_pred)
        return pred
        



    def _create_verification_sample(
        self, question: str, response: str, label: str, original_idx: int, template: str, pass_rate: float
    ) -> dict:
        """Create a verification sample dict."""
        return {
            'prompt': template.format(question=question, response=response),
            'label': label,
            'original_idx': original_idx,
            'pass_rate': pass_rate
        }

    def _fill_samples_from_pool(
        self, final_samples: list, pool_samples: list, target_size: int, with_replacement: bool = False
    ):
        """Fill final_samples from pool_samples until target_size is reached."""
        if not pool_samples:
            return
        
        needed = target_size - len(final_samples)
        if needed <= 0:
            return
        
        if with_replacement:
            final_samples.extend(random.choices(pool_samples, k=needed))
        else:
            sample_count = min(needed, len(pool_samples))
            final_samples.extend(random.sample(pool_samples, sample_count))


    def _extract_question(self, prompt: str) -> str:
        
        if 'qwen2.5' in self.tokenizer.name_or_path.lower() or 'qwen3' in self.tokenizer.name_or_path.lower():
            user_start = "<|im_start|>user\n"
            user_end = "<|im_end|>\n<|im_start|>assistant\n"
            start_idx = prompt.find(user_start)
            prompt = prompt[start_idx + len(user_start):]
            end_idx = prompt.find(user_end)
            prompt = prompt[:end_idx]
            
            if "\n\nPlease reason step by step, and put your final answer within \\boxed{}." in prompt:
                prompt = prompt.replace("\n\nPlease reason step by step, and put your final answer within \\boxed{}.", "")
            return prompt
        else:
            print(f"Unsupported tokenizer: {self.tokenizer.name_or_path}")
            return prompt


    def _add_verification_buffer(self, batch: DataProto, mode: str = "judge_answer") -> DataProto:
        """Add verification data to data buffer."""
        verification_prompt_template = """You are a teacher that is evaluating a student's answer to a question. Your task is to determine whether the answer is correct or incorrect. 

        Question: {question}
        Student's Answer: {response}

        First explain your analysis over the student's answer, the last line of your response should be of the form Answer: \\boxed{{Yes/No}} where 'Yes' means the solution is correct and 'No' means the solution is incorrect.
        \n\nPlease reason step by step, and put your final answer within \\boxed{{}}.
        """.strip()

        reward_zero_samples = []
        reward_one_samples = []
        final_reward_zero_samples = []
        final_reward_one_samples = []

        # Group samples by uid (original question)
        prompt_groups = defaultdict(list)

        batch_lens = len(batch)
        UID_list = []
        for i in range(len(batch)):
            uid = batch.non_tensor_batch["uid"][i]
            if uid not in UID_list:
                UID_list.append(uid)
            vaild = batch.non_tensor_batch['valid'][i]
            reward = batch.batch["token_level_rewards"][i].sum().item()
            if vaild == 0:
                continue
            prompt_groups[uid].append((i, reward))
        
        rollout_size = batch_lens // len(UID_list)

        target_half_size = len(prompt_groups) // 2
        self.target_size = target_half_size * 2
        remaining_groups = []
        # breakpoint()
        # Phase 1: Process all-zero and all-one groups

        All_wrong = 0
        All_right = 0
        for uid, samples in prompt_groups.items():

            # breakpoint()
            original_idx = samples[0][0]

            # Extract question (handle left-padded prompts)
            prompt_ids = batch.batch["prompts"][original_idx]
            attention_mask = batch.batch["attention_mask"][original_idx]
            prompt_mask = attention_mask[:len(prompt_ids)]
            if isinstance(prompt_mask, torch.Tensor):
                nonzero_indices = torch.nonzero(prompt_mask, as_tuple=False)
                first_valid_idx = nonzero_indices[0][0].item() if len(nonzero_indices) > 0 else 0
            valid_prompt_ids = prompt_ids[first_valid_idx:]
            question = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

            question = self._extract_question(question)

            group_rewards = [reward >= 0.5 for _, reward in samples] # [1,1,1,0,0,0,0,1]

            pass_rate = sum(group_rewards) / rollout_size


            if (not all(r == 0 for r in group_rewards)) and (not all(r == 1 for r in group_rewards)):

                                     
                postive_samples = [(idx, r) for idx, r in samples if r >= 0.5]
                negative_samples = [(idx, r) for idx, r in samples if r < 0.5]

                selected_positive_idx = random.randrange(len(postive_samples))
                selected_idx, _ = postive_samples.pop(selected_positive_idx)
                response = self._extract_response_text(batch, selected_idx, prompt_ids)
                answer = self._extract_answer_text(batch, selected_idx, prompt_ids)
                if mode == "judge_answer":
                    text_to_be_judged = answer
                else:
                    text_to_be_judged = response
                final_reward_one_samples.append(
                    self._create_verification_sample(question, text_to_be_judged, 'Yes', selected_idx, verification_prompt_template, pass_rate)
                )

                selected_negative_idx = random.randrange(len(negative_samples))
                selected_idx, _ = negative_samples.pop(selected_negative_idx)
                response = self._extract_response_text(batch, selected_idx, prompt_ids)
                answer = self._extract_answer_text(batch, selected_idx, prompt_ids)
                if mode == "judge_answer":
                    text_to_be_judged = answer
                else:
                    text_to_be_judged = response
                final_reward_zero_samples.append(
                    self._create_verification_sample(question, text_to_be_judged, 'No', selected_idx, verification_prompt_template, pass_rate)
                )

                if postive_samples:  # 确保还有剩余的正样本
                    selected_idx, _ = random.choice(postive_samples)
                    response = self._extract_response_text(batch, selected_idx, prompt_ids)
                    answer = self._extract_answer_text(batch, selected_idx, prompt_ids)
                    if mode == "judge_answer":
                        text_to_be_judged = answer
                    else:
                        text_to_be_judged = response        
                    reward_one_samples.append(
                        self._create_verification_sample(question, text_to_be_judged, 'Yes', selected_idx, verification_prompt_template, pass_rate)
                    )

                if negative_samples:  # 确保还有剩余的负样本
                    selected_idx, _ = random.choice(negative_samples)
                    response = self._extract_response_text(batch, selected_idx, prompt_ids)
                    answer = self._extract_answer_text(batch, selected_idx, prompt_ids)
                    if mode == "judge_answer":
                        text_to_be_judged = answer
                    else:
                        text_to_be_judged = response
                    reward_zero_samples.append(
                        self._create_verification_sample(question, text_to_be_judged, 'No', selected_idx, verification_prompt_template, pass_rate)
                    )

            elif all(r == 0 for r in group_rewards):
                All_wrong += 1
                # selected_idx, _ = random.choice(samples)
                continue

            elif all(r == 1 for r in group_rewards):
                All_right += 1
                selected_idx, _ = random.choice(samples)
                response = self._extract_response_text(batch, selected_idx, prompt_ids)
                answer = self._extract_answer_text(batch, selected_idx, prompt_ids)
                if mode == "judge_answer":
                    text_to_be_judged = answer
                else:
                    text_to_be_judged = response
                reward_one_samples.append(
                    self._create_verification_sample(question, text_to_be_judged, 'Yes', selected_idx, verification_prompt_template, pass_rate)
                )

            else:
                remaining_groups.append((uid, samples, question, prompt_ids))
    
        print(f"All_ERROR Rate: {All_wrong / len(prompt_groups)}")
        print(f"All_RIGHT Rate: {All_right / len(prompt_groups)}")

        # Phase 3: Fill final collections from pools (without replacement first, then with replacement)
        self._fill_samples_from_pool(final_reward_zero_samples, reward_zero_samples, target_half_size)
        self._fill_samples_from_pool(final_reward_one_samples, reward_one_samples, target_half_size)
        self._fill_samples_from_pool(final_reward_zero_samples, reward_zero_samples, target_half_size, with_replacement=True)
        self._fill_samples_from_pool(final_reward_one_samples, reward_one_samples, target_half_size, with_replacement=True)

        print(f"Verification samples: zero={len(final_reward_zero_samples)}, one={len(final_reward_one_samples)}, target={target_half_size}")

        # Combine and shuffle
        verification_data_samples = (
            final_reward_zero_samples[:target_half_size] + final_reward_one_samples[:target_half_size]
        )
        random.shuffle(verification_data_samples)
        
        self.verification_buffer.extend(verification_data_samples)








    def _create_verification_prompts(self, batch: DataProto) -> DataProto:
        """Create verification prompts based on the rewards of initial responses."""

        verification_prompt_template = """You are a teacher that is evaluating a student's answer to a question. Your task is to determine whether the answer is correct or incorrect. 

        Question: {question}
        Student's Answer: {response}

        First explain your analysis over the student's answer, the last line of your response should be of the form Answer: \\boxed{{Yes/No}} where 'Yes' means the solution is correct and 'No' means the solution is incorrect.
        \n\nPlease reason step by step, and put your final answer within \\boxed{{}}.
        """.strip()


        reward_zero_samples = []
        reward_one_samples = []
        final_reward_zero_samples = []
        final_reward_one_samples = []

        # Group samples by uid (original question)
        prompt_groups = defaultdict(list)

        batch_lens = len(batch)
        UID_list = []
        for i in range(len(batch)):
            uid = batch.non_tensor_batch["uid"][i]
            if uid not in UID_list:
                UID_list.append(uid)
            vaild = batch.non_tensor_batch['valid'][i]
            reward = batch.batch["token_level_rewards"][i].sum().item()
            if vaild == 0:
                continue
            prompt_groups[uid].append((i, reward))
        
        rollout_size = batch_lens // len(UID_list)

        target_half_size = len(prompt_groups) // 2

        self.target_size = target_half_size * 2

        
        remaining_groups = []
        # breakpoint()
        # Phase 1: Process all-zero and all-one groups

        All_wrong = 0
        All_right = 0
        for uid, samples in prompt_groups.items():

            # breakpoint()
            original_idx = samples[0][0]

            # Extract question (handle left-padded prompts)
            prompt_ids = batch.batch["prompts"][original_idx]
            attention_mask = batch.batch["attention_mask"][original_idx]
            prompt_mask = attention_mask[:len(prompt_ids)]
            if isinstance(prompt_mask, torch.Tensor):
                nonzero_indices = torch.nonzero(prompt_mask, as_tuple=False)
                first_valid_idx = nonzero_indices[0][0].item() if len(nonzero_indices) > 0 else 0
            valid_prompt_ids = prompt_ids[first_valid_idx:]
            question = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

            question = self._extract_question(question)

            group_rewards = [reward >= 0.5 for _, reward in samples] # [1,1,1,0,0,0,0,1]

            pass_rate = sum(group_rewards) / rollout_size


            if (not all(r == 0 for r in group_rewards)) and (not all(r == 1 for r in group_rewards)):

                                     
                postive_samples = [(idx, r) for idx, r in samples if r >= 0.5]
                negative_samples = [(idx, r) for idx, r in samples if r < 0.5]

                selected_positive_idx = random.randrange(len(postive_samples))
                selected_idx, _ = postive_samples.pop(selected_positive_idx)
                response = self._extract_response_text(batch, selected_idx, prompt_ids)
                final_reward_one_samples.append(
                    self._create_verification_sample(question, response, 'Yes', selected_idx, verification_prompt_template, pass_rate)
                )

                selected_negative_idx = random.randrange(len(negative_samples))
                selected_idx, _ = negative_samples.pop(selected_negative_idx)
                response = self._extract_response_text(batch, selected_idx, prompt_ids)
                final_reward_zero_samples.append(
                    self._create_verification_sample(question, response, 'No', selected_idx, verification_prompt_template, pass_rate)
                )

                if postive_samples:  # 确保还有剩余的正样本
                    selected_idx, _ = random.choice(postive_samples)
                    response = self._extract_response_text(batch, selected_idx, prompt_ids)
                    reward_one_samples.append(
                        self._create_verification_sample(question, response, 'Yes', selected_idx, verification_prompt_template, pass_rate)
                    )

                if negative_samples:  # 确保还有剩余的负样本
                    selected_idx, _ = random.choice(negative_samples)
                    response = self._extract_response_text(batch, selected_idx, prompt_ids)
                    reward_zero_samples.append(
                        self._create_verification_sample(question, response, 'No', selected_idx, verification_prompt_template, pass_rate)
                    )

            elif all(r == 0 for r in group_rewards):
                All_wrong += 1
                # selected_idx, _ = random.choice(samples)
                continue

            elif all(r == 1 for r in group_rewards):
                All_right += 1
                selected_idx, _ = random.choice(samples)
                response = self._extract_response_text(batch, selected_idx, prompt_ids)
                reward_one_samples.append(
                    self._create_verification_sample(question, response, 'Yes', selected_idx, verification_prompt_template, pass_rate)
                )

            else:
                remaining_groups.append((uid, samples, question, prompt_ids))
    
        print(f"All_ERROR Rate: {All_wrong / len(prompt_groups)}")
        print(f"All_RIGHT Rate: {All_right / len(prompt_groups)}")

        # Phase 3: Fill final collections from pools (without replacement first, then with replacement)
        self._fill_samples_from_pool(final_reward_zero_samples, reward_zero_samples, target_half_size)
        self._fill_samples_from_pool(final_reward_one_samples, reward_one_samples, target_half_size)
        self._fill_samples_from_pool(final_reward_zero_samples, reward_zero_samples, target_half_size, with_replacement=True)
        self._fill_samples_from_pool(final_reward_one_samples, reward_one_samples, target_half_size, with_replacement=True)

        print(f"Verification samples: zero={len(final_reward_zero_samples)}, one={len(final_reward_one_samples)}, target={target_half_size}")

        # Combine and shuffle
        verification_data_samples = (
            final_reward_zero_samples[:target_half_size] + final_reward_one_samples[:target_half_size]
        )
        random.shuffle(verification_data_samples)

        if not verification_data_samples:
            return None

        # Tokenize verification prompts
        verification_batch_data = []
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        for sample in verification_data_samples:
            messages = [{"role": "user", "content": sample['prompt']}]

            if hasattr(self, 'processor') and self.processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs
                )
                model_inputs = self.processor(raw_prompt, return_tensors="pt", add_special_tokens=False)
            else:
                if apply_chat_template_kwargs.get("chat_template") is None:
                    assert hasattr(self.tokenizer, "chat_template"), (
                        "chat_template should be provided in apply_chat_template_kwargs or tokenizer config"
                    )
                raw_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs
                )
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

            input_ids = model_inputs["input_ids"].squeeze(0)
            attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)

            verification_batch_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': sample['label'],
                'original_idx': sample['original_idx'],
                'pass_rate': sample['pass_rate']
            })

        # Create padded tensors (left padding)
        max_len = max(len(item['input_ids']) for item in verification_batch_data)
        batch_size = len(verification_batch_data)

        # Use pad_token_id for padding instead of 0 to avoid decoding issues
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        verification_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        verification_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        verification_position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        non_tensor_batch = {'label': [], 'original_idx': [], 'pass_rate': []}

        for i, item in enumerate(verification_batch_data):
            seq_len = len(item['input_ids'])
            start_idx = max_len - seq_len  # Left padding
            verification_input_ids[i, start_idx:] = item['input_ids']
            verification_attention_mask[i, start_idx:] = item['attention_mask']
            verification_position_ids[i, start_idx:] = torch.arange(seq_len, dtype=torch.long)
            non_tensor_batch['label'].append(item['label'])
            non_tensor_batch['original_idx'].append(item['original_idx'])
            non_tensor_batch['pass_rate'].append(item['pass_rate'])
        
        verification_batch = DataProto.from_dict(
            tensors={
                'input_ids': verification_input_ids,
                'attention_mask': verification_attention_mask,
                'position_ids': verification_position_ids
            },
            non_tensors=non_tensor_batch
        )

        verification_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )

        n = self.config.actor_rollout_ref.rollout.n
        verification_batch = verification_batch.repeat(repeat_times=n, interleave=True)

        return verification_batch


    def _verification_prompt_to_batch(self, verification_data_samples: list) -> DataProto:
        
        """Convert verification prompts to batch data."""
        verification_batch_data = []
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        for sample in verification_data_samples:
            messages = [{"role": "user", "content": sample['prompt']}]

            if hasattr(self, 'processor') and self.processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs
                )
                model_inputs = self.processor(raw_prompt, return_tensors="pt", add_special_tokens=False)
            else:
                if apply_chat_template_kwargs.get("chat_template") is None:
                    assert hasattr(self.tokenizer, "chat_template"), (
                        "chat_template should be provided in apply_chat_template_kwargs or tokenizer config"
                    )
                raw_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs
                )
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

            input_ids = model_inputs["input_ids"].squeeze(0)
            attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)

            verification_batch_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': sample['label'],
                'original_idx': sample['original_idx'],
                'pass_rate': sample['pass_rate']
            })

        # Create padded tensors (left padding)
        max_len = max(len(item['input_ids']) for item in verification_batch_data)
        batch_size = len(verification_batch_data)

        # Use pad_token_id for padding instead of 0 to avoid decoding issues
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        verification_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        verification_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        verification_position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        non_tensor_batch = {'label': [], 'original_idx': [], 'pass_rate': []}

        for i, item in enumerate(verification_batch_data):
            seq_len = len(item['input_ids'])
            start_idx = max_len - seq_len  # Left padding
            verification_input_ids[i, start_idx:] = item['input_ids']
            verification_attention_mask[i, start_idx:] = item['attention_mask']
            verification_position_ids[i, start_idx:] = torch.arange(seq_len, dtype=torch.long)
            non_tensor_batch['label'].append(item['label'])
            non_tensor_batch['original_idx'].append(item['original_idx'])
            non_tensor_batch['pass_rate'].append(item['pass_rate'])
        
        verification_batch = DataProto.from_dict(
            tensors={
                'input_ids': verification_input_ids,
                'attention_mask': verification_attention_mask,
                'position_ids': verification_position_ids
            },
            non_tensors=non_tensor_batch
        )

        verification_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )

        n = self.config.actor_rollout_ref.rollout.n
        verification_batch = verification_batch.repeat(repeat_times=n, interleave=True)
        return verification_batch        



    def _compute_generation_metrics(self, generation_batch: DataProto, timing_raw: dict) -> dict[str, Any]:
        """Compute metrics for generation task without prefix (aligned with dapo_ray_trainer.py)."""
        from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics

        generation_metrics = {}

        # Compute data metrics for generation batch (no prefix, aligned with dapo_ray_trainer.py)
        generation_data_metrics = compute_data_metrics(batch=generation_batch, use_critic=self.use_critic)
        # Use original metric names without prefix
        generation_metrics.update(generation_data_metrics)

        # Compute timing metrics for generation operations
        generation_timing_metrics = compute_timing_metrics(batch=generation_batch, timing_raw=timing_raw)
        # Filter timing metrics related to generation (exclude verification ones)
        generation_timing_keys = [k for k in generation_timing_metrics.keys()
                                 if 'verification' not in k.lower()]
        for key in generation_timing_keys:
            # Use original metric names without prefix
            generation_metrics[key] = generation_timing_metrics[key]

        # Compute throughput metrics for generation
        n_gpus = self.resource_pool_manager.get_n_gpus()
        generation_throughput_metrics = compute_throughout_metrics(
            batch=generation_batch, timing_raw=timing_raw, n_gpus=n_gpus
        )
        # Use original metric names without prefix
        generation_metrics.update(generation_throughput_metrics)

        return generation_metrics

    def _compute_verification_metrics(self, verification_batch: DataProto, timing_raw: dict) -> dict[str, Any]:
        """Compute metrics for verification task, similar to generate task."""
        from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics

        verification_metrics = {}

        # Compute data metrics for verification batch
        verification_data_metrics = compute_data_metrics(batch=verification_batch, use_critic=self.use_critic)
        # Add verification prefix to all data metrics
        for key, value in verification_data_metrics.items():
            verification_metrics[f"verification_{key}"] = value

        # Compute timing metrics for verification operations
        verification_timing_metrics = compute_timing_metrics(batch=verification_batch, timing_raw=timing_raw)
        # Filter timing metrics related to verification
        verification_timing_keys = [k for k in verification_timing_metrics.keys() if 'verification' in k.lower()]
        for key in verification_timing_keys:
            verification_metrics[key] = verification_timing_metrics[key]

        # Compute throughput metrics for verification
        n_gpus = self.resource_pool_manager.get_n_gpus()
        verification_throughput_metrics = compute_throughout_metrics(
            batch=verification_batch, timing_raw=timing_raw, n_gpus=n_gpus
        )
        # Add verification prefix to throughput metrics
        for key, value in verification_throughput_metrics.items():
            verification_metrics[f"verification_{key}"] = value

        # Add verification-specific metrics
        if "verification_correct" in verification_batch.non_tensor_batch:
            verification_correct = np.array(verification_batch.non_tensor_batch["verification_correct"])
            verification_accuracy = np.mean(verification_correct)
            verification_metrics["verification/verification_accuracy"] = verification_accuracy

        return verification_metrics

    def _train_verification_task(self, verification_batch_for_reward: DataProto, metrics: dict, timing_raw: dict):
        """Train the verification task after standard DAPO training."""
        # Apply the same training logic as standard DAPO but for verification data

        # Balance batch
        if self.config.trainer.balance_batch:
            self._balance_batch(verification_batch_for_reward, metrics=metrics)

        # Compute global token num
        verification_batch_for_reward.meta_info["global_token_num"] = torch.sum(
            verification_batch_for_reward.batch["attention_mask"], dim=-1
        ).tolist()

        # Compute KL related metrics for verification batch
        verification_batch_for_reward = self.compute_kl_related_metrics_for_verification(verification_batch_for_reward, metrics, timing_raw)

        # Compute values for verification batch
        if self.use_critic:
            with marked_timer("values_verification", timing_raw, "cyan"):
                values = self.critic_wg.compute_values(verification_batch_for_reward)
                verification_batch_for_reward = verification_batch_for_reward.union(values)

        # Compute rollout correction for verification batch
        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None and "rollout_log_probs" in verification_batch_for_reward.batch:
            verification_batch_for_reward, is_metrics = compute_rollout_correction_and_add_to_batch(
                verification_batch_for_reward, rollout_corr_config
            )
            metrics.update({f"verification_{k}": v for k, v in is_metrics.items()})

        # Compute advantages for verification batch
        with marked_timer("adv_verification", timing_raw, "brown"):
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            # ! New Algoritm
            verification_batch_for_reward = compute_advantage(
                verification_batch_for_reward,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,  # Verification has 1 response per prompt
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

            # verification_batch_for_reward = compute_advantage(
            #     verification_batch_for_reward,
            #     adv_estimator="verify_scale",
            #     gamma=self.config.algorithm.gamma,
            #     lam=self.config.algorithm.lam,
            #     num_repeat=self.config.actor_rollout_ref.rollout.n,  # Verification has 1 response per prompt
            #     norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            # )

        # Update critic on verification data
        if self.use_critic:
            with marked_timer("update_critic_verification", timing_raw, "pink"):
                critic_output = self.critic_wg.update_critic(verification_batch_for_reward)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update({f"verification_{k}": v for k, v in critic_output_metrics.items()})

        # Update actor on verification data (if critic warmup is done)
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor_verification", timing_raw, "red"):
                actor_output = self.actor_rollout_wg.update_actor(verification_batch_for_reward)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update({f"verification_{k}": v for k, v in actor_output_metrics.items()})

        # Compute and add verification-specific metrics
        verification_metrics = self._compute_verification_metrics(verification_batch_for_reward, timing_raw)
        metrics.update(verification_metrics)

    def fit(self):
        """
        The training loop of Self-Verification DAPO.
        First does standard DAPO training, then adds verification training.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        # Store the current batch for verification training
        current_generation_batch = None
        # breakpoint()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ===== STEP 1: Rollout generate task =====
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # ===== STEP 2: Train generate task =====
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Store the current generation batch for verification training
                    current_generation_batch = batch

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                # breakpoint()

                print(f"Generation Done, Starting Verification...")
                print(f"Sample generation data: {self.tokenizer.decode(current_generation_batch.batch['input_ids'][0], skip_special_tokens=True)}")
                # ===== STEP 3 & 4: Verification rollout and training =====

                # breakpoint()
                # breakpoint()
                #? collect verification buffer.
                if self.global_steps % self.config.algorithm.verification_interval == 0:
                    
                    print(f"Verification interval reached. Starting verification...")
                    #! (wy) Fix WandB issue.
                    if len(self.verification_buffer) > 0:
                    # if len(self.verification_buffer) == 0:
                    #     continue
                        verification_loop = 0
                        random.shuffle(self.verification_buffer)
                        for i in range(0, len(self.verification_buffer), self.target_size):
                            if verification_loop >= self.config.algorithm.verification_looplen:
                                break
                            verification_batch = self.verification_buffer[i:i+self.target_size]

                            verification_batch = self._verification_prompt_to_batch(verification_batch)
                            if verification_batch is not None:
                                with marked_timer("verification_rollout", timing_raw, "green"):
                                    verification_output = self.actor_rollout_wg.generate_sequences(verification_batch)
                                    verification_batch_for_reward = deepcopy(verification_output)


                                    batch_size = len(verification_batch_for_reward)
                                    verification_labels = verification_batch.non_tensor_batch["label"]

                                    reward_model_data = []
                                    for label in verification_labels:
                                        reward_model_data.append({
                                            "ground_truth": label,
                                            "style": "function"
                                        })

                                    verification_batch_for_reward.non_tensor_batch["reward_model"] = np.array(
                                        reward_model_data, dtype=object
                                    )
                                    verification_batch_for_reward.non_tensor_batch["data_source"] = np.array(
                                        ["verification"] * batch_size, dtype=object
                                    )
                                    verification_batch_for_reward.non_tensor_batch["is_verification"] = np.ones(
                                        batch_size, dtype=bool
                                    )
                                    verification_batch_for_reward.non_tensor_batch["label"] = np.array(
                                        verification_labels
                                    )
                                    if self.use_rm and "rm_scores" not in verification_batch_for_reward.batch.keys():
                                        verification_rm_scores = self.rm_wg.compute_rm_score(verification_batch_for_reward)
                                        verification_batch_for_reward = verification_batch_for_reward.union(verification_rm_scores)
                                    verification_reward_tensor, verification_reward_extra_infos = compute_reward(
                                    verification_batch_for_reward, self.reward_fn
                                    )

                                    verification_correct = []

                                    acc_list = verification_reward_extra_infos['acc']
                                    for acc_ in acc_list:
                                        verification_correct.append(acc_)


                                    verification_batch_for_reward.non_tensor_batch["verification_correct"] = np.array(verification_correct)
                                    verification_batch_for_reward.batch["token_level_scores"] = verification_reward_tensor
                                    verification_batch_for_reward.batch["token_level_rewards"] = verification_reward_tensor

                                with marked_timer("train_verification", timing_raw, "yellow"):
                                    self._train_verification_task(verification_batch_for_reward, metrics, timing_raw)

                            verification_loop += 1
                        print(f"Verification Done...")
                        print(f"Sample verification data: {self.tokenizer.decode(verification_batch_for_reward.batch['input_ids'][0], skip_special_tokens=True)}")
                        # ! flush verification buffer.
                        self.verification_buffer = []
                    else:
                        print(f"Warning: Verification buffer is empty, skipping verification training.")

                    if current_generation_batch is not None:
                        self._add_verification_buffer(current_generation_batch)

                else:
                    if current_generation_batch is not None:
                        self._add_verification_buffer(current_generation_batch)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect generation metrics (no prefix, aligned with dapo_ray_trainer.py)
                generation_metrics = self._compute_generation_metrics(batch, timing_raw)
                metrics.update(generation_metrics)

                # Reset timing for next step
                timing_raw = defaultdict(float)

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)