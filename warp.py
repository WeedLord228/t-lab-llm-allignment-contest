import torch
from transformers import Trainer
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader


class WARP():
    def __init__(
            self,
            policy_model: nn.Module,
            policy_model_tokenizer,
            # ref_model: nn.Module,
            reward_model: nn.Module,
            reward_model_tokenizer,
            prompt_dataloader: DataLoader,
            batch_size = 32,
            iter_n=2,
            rl_runs_n=2, # Only 2 supported
            training_steps_n=100,
            mu=0.01,
            beta=0.1,
            etha=0.5
    ):
        self.policy_model = policy_model
        self.policy_model_tokenizer = policy_model_tokenizer
        self.ref_model = policy_model.copy()

        self.reward_model = reward_model
        self.reward_model_tokenizer = reward_model_tokenizer

        self.prompt_dataloader = prompt_dataloader
        self.batch_size = batch_size

        self.iter_n = iter_n
        self.rl_runs_n = rl_runs_n
        self.training_steps_n = training_steps_n
        self.mu = mu
        self.beta = beta,
        self.etha = etha

    def train(self):
        init_model = self.policy_model

        # def repeat_generator():
        #     while True:
        #         yield from self.prompt_dataloader
        #
        # iter_dataloader = iter(repeat_generator())

        for i in range(self.iter_n):
            # <editor-fold desc="RL run">
            run_models = []
            for m in range(self.rl_runs_n):
                cur_run_model = init_model.copy()
                cur_run_ref_model = init_model.copy()
                for t in range(self.training_steps_n):
                    for prompts in self.prompt_dataloader:
                        # <editor-fold desc="training_step">
                        # <editor-fold desc="get completions">
                        completions = cur_run_model(prompts)
                        # </editor-fold>
                        # <editor-fold desc="get beta reward">
                        KL_part = 0
                        rewards_beta = self.reward_model(completions) - self.beta * KL_part
                        # </editor-fold>
                        # <editor-fold desc="update models">
                        update_cur = 0
                        # update EMA
                        cur_run_ref_model.params = (1 - self.mu) * cur_run_ref_model.params + self.mu * \
                                                   cur_run_model.params
                        # </editor-fold>
                        # </editor-fold>
                run_models.append(cur_run_model)

            # </editor-fold>
            # <editor-fold desc="get_slerp_weights">
            slerp_init_weights = self.slerp_two_tensors(init_model.params, run_models[0].params, run_models[1].params)
            # </editor-fold>
            # <editor-fold desc="update init">
            init_params = (1 - self.etha) * init_model.params + self.etha * slerp_init_weights
            # </editor-fold>

    def slerp_two_tensors(
            self,
            weights_init: torch.Tensor,
            weights_a: torch.Tensor,
            weights_b: torch.Tensor
    ):
        inter_coeff = 1 / self.rl_runs_n  # lambda
        delta_a = weights_a - weights_init
        delta_b = weights_b - weights_init
        omega = torch.arccos(torch.dot(delta_a / delta_a.norm(), delta_b / delta_b.norm()))
        so = torch.sin(omega)

        result = weights_init + \
                 torch.sin((1 - inter_coeff) * omega) * delta_a / so + \
                 torch.sin(inter_coeff * omega) / so * delta_b

        return result
