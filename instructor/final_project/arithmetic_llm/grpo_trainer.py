"""GRPO trainer module."""

from typing import Any, Dict, List, Optional, Tuple

import math
import json
import os
from datetime import datetime, timezone
import time

import torch
import torch.nn.functional as F

from .arithmetic_tokenizer import ArithmeticBPETokenizer
from .arithmetic_verifier import ArithmeticVerifier
from .train_foundational import (
    get_linear_schedule_with_warmup,
    load_checkpoint,
)
from .transformer_model import ArithmeticTransformer

from .grpo_config import GRPOConfig


class GRPOTrainer:
    """GRPO trainer for arithmetic LLM."""

    def __init__(
        self,
        config: GRPOConfig,
        sft_checkpoint_path: Optional[str] = None,
        tokenizer: Optional[ArithmeticBPETokenizer] = None,
        tokenizer_path: Optional[str] = None,
        policy_model: Optional[torch.nn.Module] = None,
        reference_model: Optional[torch.nn.Module] = None,
        total_steps: Optional[int] = None,
        use_mixed_precision: bool = False,
        candidate_sub_batch_size: Optional[int] = None
    ):
        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.optimizer = None
        self.scheduler = None
        self.verifier = ArithmeticVerifier()
        self.use_mixed_precision = use_mixed_precision
        self.candidate_sub_batch_size = candidate_sub_batch_size
        device_type = "cuda" if self.config.device == "cuda" else "cpu"
        try:
            self._scaler = torch.amp.GradScaler(
                device_type=device_type,
                enabled=use_mixed_precision
            )
        except TypeError:
            if device_type == "cuda":
                self._scaler = torch.cuda.amp.GradScaler(
                    enabled=use_mixed_precision
                )
            else:
                self._scaler = torch.amp.GradScaler(enabled=use_mixed_precision)

        if self.tokenizer is None and tokenizer_path is not None:
            self.tokenizer = ArithmeticBPETokenizer()
            self.tokenizer.load(tokenizer_path)

        if self.policy_model is None or self.reference_model is None:
            if sft_checkpoint_path is not None:
                self._load_models_from_checkpoint(sft_checkpoint_path)

        if self.reference_model is not None:
            self._freeze_reference_model()

        if self.policy_model is not None:
            self.policy_model = self.policy_model.to(self.config.device)
            params = list(self.policy_model.parameters())
            if params:
                self.optimizer = torch.optim.AdamW(
                    params,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.01
                )
                if total_steps is not None:
                    self.scheduler = get_linear_schedule_with_warmup(
                        optimizer=self.optimizer,
                        num_warmup_steps=self.config.warmup_steps,
                        num_training_steps=max(1, total_steps)
                    )

        if self.reference_model is not None:
            self.reference_model = self.reference_model.to(self.config.device)

    def _forward_model(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward helper that tolerates models without attention_mask support."""
        if attention_mask is None:
            return model(input_ids)
        try:
            return model(input_ids, attention_mask=attention_mask)
        except TypeError:
            return model(input_ids)

    def _require_generation_components(self) -> None:
        if self.policy_model is None or self.tokenizer is None:
            raise ValueError(
                "policy_model and tokenizer must be provided to generate candidates"
            )

    def _load_models_from_checkpoint(self, checkpoint_path: str) -> None:
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided to load models")

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        vocab_size = len(self.tokenizer.token2id)
        model_config = checkpoint_data.get(
            "model_config",
            {
                "vocab_size": vocab_size,
                "d_model": 256,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 1024,
                "dropout": 0.1,
                "max_seq_length": 512,
            },
        )
        checkpoint_vocab_size = checkpoint_data.get("tokenizer_vocab_size")
        if checkpoint_vocab_size is not None and checkpoint_vocab_size != vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({vocab_size}) does not match checkpoint "
                f"vocab size ({checkpoint_vocab_size})."
            )

        model_config["vocab_size"] = vocab_size
        self.policy_model = ArithmeticTransformer(**model_config)
        self.reference_model = ArithmeticTransformer(**model_config)

        load_checkpoint(checkpoint_path=checkpoint_path, model=self.policy_model)
        load_checkpoint(checkpoint_path=checkpoint_path, model=self.reference_model)

    def reset_optimizer_and_scheduler(self, total_steps: Optional[int] = None) -> None:
        if self.policy_model is None:
            raise ValueError("policy_model must be initialized")
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        if total_steps is not None:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=max(1, total_steps)
            )

    def memory_usage_estimate(
        self,
        batch_size: int,
        num_candidates: int,
        max_gen_length: int
    ) -> Dict[str, int]:
        """Return rough memory usage estimates in bytes."""
        if self.policy_model is None:
            return {"parameter_bytes": 0, "activation_bytes": 0, "total_bytes": 0}
        param_bytes = sum(p.numel() * p.element_size() for p in self.policy_model.parameters())
        activation_bytes = batch_size * num_candidates * max_gen_length * 4
        return {
            "parameter_bytes": int(param_bytes),
            "activation_bytes": int(activation_bytes),
            "total_bytes": int(param_bytes + activation_bytes),
        }

    def _freeze_reference_model(self) -> None:
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_group_statistics(
        self,
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute group mean and standard deviation."""
        group_mean = torch.mean(rewards, dim=1)
        group_std = torch.stack(
            [torch.std(row, unbiased=False) for row in rewards.unbind(dim=0)]
        )
        return group_mean, group_std

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages."""
        group_mean, group_std = self.compute_group_statistics(rewards)
        group_mean = group_mean.unsqueeze(1)
        group_std = group_std.unsqueeze(1)
        return (rewards - group_mean) / (group_std + self.config.advantage_epsilon)

    def normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Normalize advantages to zero mean within each group."""
        mean = torch.mean(advantages, dim=1, keepdim=True)
        return advantages - mean

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy gradient loss."""
        if log_probs.shape != advantages.shape:
            raise ValueError(
                "log_probs and advantages must have the same shape, got "
                f"{log_probs.shape} and {advantages.shape}"
            )
        return -torch.mean(advantages * log_probs)

    def compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference logits."""
        if policy_logits.shape != reference_logits.shape:
            raise ValueError(
                "policy_logits and reference_logits must have the same shape, got "
                f"{policy_logits.shape} and {reference_logits.shape}"
            )
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        policy_probs = torch.softmax(policy_logits, dim=-1)
        kl = policy_probs * (policy_log_probs - reference_log_probs)
        kl_sum = torch.sum(kl, dim=-1)
        return torch.mean(kl_sum)

    def compute_total_loss(
        self,
        policy_loss: torch.Tensor,
        kl_divergence: torch.Tensor,
        kl_penalty_coef: Optional[float] = None
    ) -> torch.Tensor:
        """Compute total loss including KL penalty."""
        coef = self.config.kl_penalty_coef if kl_penalty_coef is None else kl_penalty_coef
        policy_loss = policy_loss.float()
        kl_divergence = kl_divergence.float()
        coef_tensor = torch.tensor(
            coef, device=kl_divergence.device, dtype=kl_divergence.dtype
        )
        return policy_loss + coef_tensor * kl_divergence

    def train_step(
        self,
        prompts: List[str],
        ground_truth: List[int],
        do_step: bool = True,
        loss_scale: float = 1.0
    ) -> dict:
        """Execute single GRPO training step."""
        if len(prompts) != len(ground_truth):
            raise ValueError("prompts and ground_truth must have the same length")

        if self.policy_model is None or self.reference_model is None:
            raise ValueError("policy_model and reference_model must be initialized")
        if self.optimizer is None:
            raise ValueError("optimizer must be initialized for training")

        self._require_generation_components()
        self.policy_model.train()

        num_candidates = self.config.num_candidates
        generated_texts, _ = self.generate_candidates(
            prompts, num_candidates=num_candidates
        )

        device = next(self.policy_model.parameters()).device
        bos_id = self.tokenizer.token2id.get("<bos>", 0)
        pad_id = self.tokenizer.token2id.get("<pad>", 0)

        rewards_list: List[float] = []
        log_probs_list: List[torch.Tensor] = []
        kl_list: List[torch.Tensor] = []
        flat_generated_ids: List[List[int]] = []
        flat_prompt_lens: List[int] = []

        for prompt_idx, prompt in enumerate(prompts):
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = 1 + len(prompt_tokens)

            for cand_idx in range(num_candidates):
                text = generated_texts[prompt_idx][cand_idx]
                reward = self.verifier.compute_reward(text, ground_truth[prompt_idx])
                rewards_list.append(float(reward))

                generated_ids = self.tokenizer.encode(text, add_special_tokens=True)
                flat_generated_ids.append(generated_ids)
                flat_prompt_lens.append(prompt_len)

        total_candidates = len(flat_generated_ids)
        sub_batch = self.candidate_sub_batch_size or total_candidates

        device_type = "cuda" if self.config.device == "cuda" else "cpu"
        positions_cache = {}

        for start in range(0, total_candidates, sub_batch):
            end = min(start + sub_batch, total_candidates)
            chunk_ids = flat_generated_ids[start:end]
            chunk_prompt_lens = flat_prompt_lens[start:end]
            max_len = max(len(ids) for ids in chunk_ids)

            padded_input_ids = []
            attention_masks = []
            pad_lens = []
            for ids in chunk_ids:
                pad_len = max_len - len(ids)
                pad_lens.append(pad_len)
                # Right-pad to keep prompt positions consistent with training.
                padded_input_ids.append(ids + [pad_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)

            input_tensor = torch.tensor(
                padded_input_ids, dtype=torch.long, device=device
            )
            attention_mask = torch.tensor(
                attention_masks, dtype=torch.float32, device=device
            )

            with torch.amp.autocast(
                device_type=device_type,
                enabled=self.use_mixed_precision
            ):
                policy_logits = self._forward_model(
                    self.policy_model, input_tensor, attention_mask=attention_mask
                )
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self.use_mixed_precision
                ):
                    reference_logits = self._forward_model(
                        self.reference_model, input_tensor, attention_mask=attention_mask
                    )

            log_probs = torch.log_softmax(policy_logits, dim=-1)
            targets = input_tensor[:, 1:]
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            valid_mask = attention_mask[:, 1:]
            positions = positions_cache.get(token_log_probs.shape[1])
            if positions is None:
                positions = torch.arange(
                    token_log_probs.shape[1], device=device
                )
                positions_cache[token_log_probs.shape[1]] = positions

            prompt_lens_tensor = torch.tensor(chunk_prompt_lens, device=device)
            # With right-padding, prompt tokens start at index 0.
            start_indices = prompt_lens_tensor - 1
            start_indices = torch.clamp(start_indices, min=0)

            start_mask = positions.unsqueeze(0) >= start_indices.unsqueeze(1)
            token_mask = (valid_mask > 0) & start_mask
            log_prob_sum = (token_log_probs * token_mask).sum(dim=1)

            policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
            reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
            policy_probs = torch.softmax(policy_logits, dim=-1)
            kl = policy_probs * (policy_log_probs - reference_log_probs)
            kl_sum = torch.sum(kl, dim=-1)
            kl_mask = attention_mask
            kl_denom = torch.clamp(kl_mask.sum(dim=1), min=1.0)
            kl_value = (kl_sum * kl_mask).sum(dim=1) / kl_denom

            log_probs_list.extend(log_prob_sum)
            kl_list.extend(kl_value)

        batch_size = len(prompts)
        rewards_tensor = torch.tensor(
            rewards_list, dtype=torch.float32, device=device
        ).view(batch_size, num_candidates)

        advantages = self.compute_advantages(rewards_tensor)
        advantages = self.normalize_advantages(advantages)

        log_probs_tensor = torch.stack(log_probs_list).view(
            batch_size, num_candidates
        )
        policy_loss = self.compute_policy_loss(log_probs_tensor, advantages)

        if kl_list:
            kl_divergence = torch.mean(torch.stack(kl_list))
        else:
            kl_divergence = torch.tensor(0.0, device=device)

        total_loss = self.compute_total_loss(policy_loss, kl_divergence)

        scaled_loss = total_loss * loss_scale
        if self.use_mixed_precision:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if do_step:
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            if self.use_mixed_precision:
                self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.gradient_clip
            )
            if self.use_mixed_precision:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()

        avg_reward = rewards_tensor.mean().item()
        reward_rate = (rewards_tensor > 0.5).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_divergence.item(),
            "total_loss": total_loss.item(),
            "avg_reward": avg_reward,
            "reward_rate": reward_rate,
        }

    def train(
        self,
        train_dataloader,
        val_dataloader: Optional[Any] = None,
        output_dir: str = "."
    ) -> Dict[str, Any]:
        """Run GRPO training loop."""
        if self.policy_model is None or self.reference_model is None:
            raise ValueError("policy_model and reference_model must be initialized")
        if self.optimizer is None:
            raise ValueError("optimizer must be initialized for training")

        os.makedirs(output_dir, exist_ok=True)
        training_log: List[Dict[str, Any]] = []
        global_step = 0
        best_reward_rate = -1.0
        accum_steps = max(1, self.config.gradient_accumulation_steps)
        total_steps = None
        try:
            total_steps = math.ceil(len(train_dataloader) / accum_steps) * self.config.num_epochs
        except TypeError:
            total_steps = None

        for epoch in range(self.config.num_epochs):
            step_start = None
            step_time = None
            accum_metrics = None
            accum_batches = 0
            for batch_idx, batch in enumerate(train_dataloader):
                did_step = False
                if isinstance(batch, dict):
                    prompts = batch.get("prompts")
                    ground_truth = batch.get("ground_truth")
                else:
                    prompts, ground_truth = batch

                if prompts is None or ground_truth is None:
                    raise ValueError("batch must contain prompts and ground_truth")

                if batch_idx % accum_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    step_start = time.perf_counter()
                    step_time = None
                    accum_metrics = None
                    accum_batches = 0

                metrics = self.train_step(
                    prompts=prompts,
                    ground_truth=ground_truth,
                    do_step=False,
                    loss_scale=1.0 / accum_steps
                )
                if accum_metrics is None:
                    accum_metrics = {key: 0.0 for key in metrics}
                for key, value in metrics.items():
                    accum_metrics[key] += float(value)
                accum_batches += 1

                val_metrics = None
                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    global_step += 1
                    did_step = True
                    if step_start is not None:
                        step_time = time.perf_counter() - step_start
                    step_start = None

                    if global_step % self.config.save_every == 0:
                        self.save_checkpoint(
                            output_dir=output_dir,
                            step=global_step,
                            epoch=epoch + 1,
                            metrics=metrics,
                            is_final=False
                        )

                    if val_dataloader is not None and global_step % self.config.eval_every == 0:
                        val_metrics = self.evaluate(val_dataloader)
                        if val_metrics["reward_rate"] > best_reward_rate:
                            best_reward_rate = val_metrics["reward_rate"]
                            best_path = self.save_checkpoint(
                                output_dir=output_dir,
                                step=global_step,
                                epoch=epoch + 1,
                                metrics=val_metrics,
                                is_final=False
                            )
                            best_model_path = os.path.join(output_dir, "best_model.pt")
                            os.replace(best_path, best_model_path)

                if self.scheduler is not None:
                    learning_rate = self.scheduler.get_last_lr()[0]
                else:
                    learning_rate = self.optimizer.param_groups[0]["lr"]

                if did_step and global_step % self.config.log_every == 0:
                    avg_metrics = metrics
                    if accum_metrics is not None and accum_batches:
                        avg_metrics = {
                            key: value / accum_batches
                            for key, value in accum_metrics.items()
                        }
                    step_label = (
                        f"{global_step}/{total_steps}"
                        if total_steps is not None
                        else f"{global_step}"
                    )
                    time_fragment = ""
                    if step_time is not None:
                        time_fragment = f"time {step_time:.2f}s "
                    print(
                        "[grpo] "
                        f"step {step_label} "
                        f"epoch {epoch + 1}/{self.config.num_epochs} "
                        f"{time_fragment}"
                        f"loss {avg_metrics['total_loss']:.4f} "
                        f"kl {avg_metrics['kl_divergence']:.4f} "
                        f"reward {avg_metrics['avg_reward']:.3f} "
                        f"reward_rate {avg_metrics['reward_rate']:.3f} "
                        f"lr {learning_rate:.2e}",
                        flush=True
                    )

                log_metrics = metrics
                if did_step and accum_metrics is not None and accum_batches:
                    log_metrics = {
                        key: value / accum_batches
                        for key, value in accum_metrics.items()
                    }

                training_log.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch_idx": batch_idx,
                    "metrics": {
                        "policy_loss": log_metrics["policy_loss"],
                        "kl_divergence": log_metrics["kl_divergence"],
                        "total_loss": log_metrics["total_loss"],
                        "avg_reward": log_metrics["avg_reward"],
                        "reward_rate": log_metrics["reward_rate"],
                        "val_reward_rate": val_metrics["reward_rate"] if val_metrics else None,
                    },
                    "learning_rate": learning_rate,
                    "step_time_s": step_time if did_step else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        final_checkpoint_path = self.save_checkpoint(
            output_dir=output_dir,
            step=global_step,
            epoch=self.config.num_epochs,
            metrics=training_log[-1]["metrics"] if training_log else {},
            is_final=True
        )

        log_path = os.path.join(output_dir, "grpo_training_log.json")
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        return {
            "global_step": global_step,
            "log_path": log_path,
            "final_checkpoint_path": final_checkpoint_path,
        }

    def save_checkpoint(
        self,
        output_dir: str,
        step: int,
        epoch: int,
        metrics: Dict[str, Any],
        is_final: bool = False
    ) -> str:
        """Save GRPO checkpoint with metadata."""
        if self.policy_model is None or self.optimizer is None:
            raise ValueError("policy_model and optimizer must be initialized")

        os.makedirs(output_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.policy_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "step": step,
            "config": self.config.to_dict(),
            "grpo_config": {
                "num_candidates": self.config.num_candidates,
                "temperature": self.config.temperature,
                "kl_penalty_coef": self.config.kl_penalty_coef,
            },
            "metrics": metrics,
        }

        if isinstance(self.policy_model, ArithmeticTransformer):
            checkpoint["model_config"] = {
                "vocab_size": self.policy_model.vocab_size,
                "d_model": self.policy_model.d_model,
                "nhead": self.policy_model.nhead,
                "num_layers": self.policy_model.num_layers,
                "dim_feedforward": self.policy_model.dim_feedforward,
                "dropout": self.policy_model.dropout,
                "max_seq_length": self.policy_model.max_seq_length,
            }

        if is_final:
            checkpoint_path = os.path.join(output_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load GRPO checkpoint and restore trainer state."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if self.policy_model is None:
            model_config = checkpoint.get("model_config")
            if model_config is None:
                raise ValueError("Checkpoint missing model_config")
            self.policy_model = ArithmeticTransformer(**model_config)
        self.policy_model.load_state_dict(checkpoint["model_state_dict"])
        self.policy_model = self.policy_model.to(self.config.device)

        if self.reference_model is None:
            self.reference_model = ArithmeticTransformer(**checkpoint["model_config"])
        self.reference_model.load_state_dict(checkpoint["model_state_dict"])
        self.reference_model = self.reference_model.to(self.config.device)
        self._freeze_reference_model()

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.policy_model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        if checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("scheduler_state_dict") and self.optimizer is not None:
            if self.scheduler is None:
                self.scheduler = get_linear_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=self.config.warmup_steps,
                    num_training_steps=1
                )
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def evaluate(self, val_dataloader) -> Dict[str, float]:
        """Run validation evaluation and return reward metrics."""
        if self.policy_model is None:
            raise ValueError("policy_model must be initialized")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be initialized")

        self.policy_model.eval()
        total = 0
        correct = 0

        for batch in val_dataloader:
            if isinstance(batch, dict):
                prompts = batch.get("prompts")
                ground_truth = batch.get("ground_truth")
            else:
                prompts, ground_truth = batch

            if prompts is None or ground_truth is None:
                raise ValueError("batch must contain prompts and ground_truth")

            generated_texts, _ = self.generate_candidates(prompts, num_candidates=1)
            for idx, prompt in enumerate(prompts):
                text = generated_texts[idx][0]
                reward = self.verifier.compute_reward(text, ground_truth[idx])
                correct += 1 if reward > 0.5 else 0
                total += 1

        reward_rate = correct / max(total, 1)
        return {
            "reward_rate": reward_rate,
            "total": total,
            "correct": correct,
        }

    def generate_candidates(
        self,
        prompts: List[str],
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_gen_length: Optional[int] = None
    ) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
        """Generate multiple candidate responses per prompt."""
        self._require_generation_components()

        if num_candidates is None:
            num_candidates = self.config.num_candidates
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p
        if max_gen_length is None:
            max_gen_length = self.config.max_gen_length

        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")

        if not prompts:
            return [], []

        first_param = next(self.policy_model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        bos_id = self.tokenizer.token2id.get("<bos>", 0)
        eos_id = self.tokenizer.token2id.get("<eos>", None)
        pad_id = self.tokenizer.token2id.get("<pad>", 0)

        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_candidates)

        prompt_tokens_list = []
        for prompt in expanded_prompts:
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tokens_list.append([bos_id] + prompt_tokens)

        max_prompt_len = max(len(ids) for ids in prompt_tokens_list)
        padded_ids = []
        attention_masks = []
        for ids in prompt_tokens_list:
            pad_len = max_prompt_len - len(ids)
            # Right-pad so positions align with instruction training.
            padded_ids.append(ids + [pad_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.float32, device=device)

        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        log_probs_lists: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        self.policy_model.eval()
        with torch.no_grad():
            while input_ids.shape[1] < max_gen_length:
                device_type = "cuda" if self.config.device == "cuda" else "cpu"
                with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self.use_mixed_precision
                ):
                    logits = self._forward_model(
                        self.policy_model, input_ids, attention_mask=attention_mask
                    )
                next_token_logits = logits[:, -1, :]

                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    kth = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    indices_to_remove = next_token_logits < kth
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                token_log_prob = torch.log(
                    probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
                )

                active = ~finished
                if active.any():
                    for idx in torch.nonzero(active, as_tuple=False).squeeze(1).tolist():
                        log_probs_lists[idx].append(token_log_prob[idx])

                if eos_id is not None:
                    just_finished = active & (next_token == eos_id)
                else:
                    just_finished = torch.zeros_like(finished)

                next_token = torch.where(active, next_token, torch.tensor(pad_id, device=device))
                next_attention = torch.where(active, torch.ones_like(next_token), torch.zeros_like(next_token))

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                attention_mask = torch.cat([attention_mask, next_attention.unsqueeze(1).float()], dim=1)

                finished = finished | just_finished
                if finished.all():
                    break

        decoded_texts = [
            self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            for ids in input_ids
        ]

        all_texts: List[List[str]] = []
        all_log_probs: List[List[torch.Tensor]] = []
        idx = 0
        for _ in prompts:
            prompt_texts = []
            prompt_log_probs = []
            for _ in range(num_candidates):
                prompt_texts.append(decoded_texts[idx])
                lp = log_probs_lists[idx]
                prompt_log_probs.append(torch.stack(lp) if lp else torch.tensor([]))
                idx += 1
            all_texts.append(prompt_texts)
            all_log_probs.append(prompt_log_probs)

        return all_texts, all_log_probs

    def compute_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of generated sequence."""
        if self.policy_model is None:
            raise ValueError("policy_model must be provided to compute log probabilities")

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if generated_ids.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)

        prompt_len = input_ids.shape[1]
        if generated_ids.shape[1] <= 1:
            return torch.tensor(0.0, device=generated_ids.device)

        logits = self.policy_model(generated_ids[:, :-1])
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = generated_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        start_index = max(prompt_len - 1, 0)
        token_log_probs = token_log_probs[:, start_index:]
        return torch.sum(token_log_probs)


