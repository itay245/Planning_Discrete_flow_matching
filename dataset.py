from typing import Sequence

import torch
from torch.utils.data import Dataset


class PlanningDFMDataset(Dataset):
    """
    Dataset containing:

        context
        context_mask
        clean target plan X_1

    Negative/non-goal trajectories are intentionally NOT included
    in vanilla DFM training.

    DFM learns the distribution of valid plans.
    """

    def __init__(
        self,
        examples: Sequence[dict],
    ):
        self.examples = list(
            examples
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(
        self,
        index,
    ):
        example = self.examples[
            index
        ]

        return {
            "context":
                example["context"],

            "context_mask":
                example["context_mask"],

            "plan":
                example["plan"],
        }

    @classmethod
    def from_single_problem(
        cls,
        problem,
        plans,
        codec,
    ):
        """
        Convenience constructor when all plans correspond to the
        same initial state / goal.
        """

        encoded_context = (
            codec.encode_problem_context(
                problem
            )
        )

        examples = []

        for plan in plans:

            encoded_plan = (
                codec.encode_plan(
                    plan
                )
            )

            examples.append(
                {
                    "context":
                        encoded_context.tensor,

                    "context_mask":
                        encoded_context.attention_mask,

                    "plan":
                        encoded_plan.tensor,
                }
            )

        return cls(examples)