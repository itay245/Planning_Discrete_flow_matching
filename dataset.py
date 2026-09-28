# dataset.py

from typing import Sequence

import torch
from torch.utils.data import Dataset


class PlanningDFMDataset(Dataset):
    """
    Dataset for conditional DFM plan generation.

    Each sample contains:

        context:
            encoded initial state + goal

        context_mask:
            attention mask for padded context rows

        plan:
            encoded clean target plan X_1

    Multiple valid plans may share exactly the same context.
    """

    def __init__(
        self,
        examples: Sequence[dict],
    ):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        return {
            "context": example["context"],
            "context_mask": example["context_mask"],
            "plan": example["plan"],
        }

    # ========================================================
    # Single problem
    # ========================================================

    @classmethod
    def from_single_problem(
        cls,
        problem,
        plans,
        codec,
    ):
        """
        Create a dataset from one planning problem and all of
        its valid plans.

        Every plan receives the same initial-state/goal context.
        """

        encoded_context = codec.encode_problem_context(
            problem
        )

        examples = []

        for plan in plans:

            encoded_plan = codec.encode_plan(
                plan
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

    # ========================================================
    # Multiple problems
    # ========================================================

    @classmethod
    def from_problems(
        cls,
        problem_plan_pairs,
        codec,
    ):
        """
        Construct one dataset from multiple planning problems.

        Parameters
        ----------
        problem_plan_pairs:

            Iterable of:

                (problem, plans)

            where `plans` is a collection of valid symbolic plans
            for that problem.

        Example:

            [
                (problem_1, plans_1),
                (problem_2, plans_2),
                (problem_3, plans_3),
            ]

        codec:

            A single shared PlanCodec.

            This assumes that all problems belong to the same
            planning domain and use a compatible object/vocabulary
            structure.
        """

        examples = []

        expected_context_shape = None

        for problem_index, (
            problem,
            plans,
        ) in enumerate(problem_plan_pairs):

            # ---------------------------------------------
            # Encode this problem's s0 and goal once.
            # ---------------------------------------------

            encoded_context = (
                codec.encode_problem_context(
                    problem
                )
            )

            # ---------------------------------------------
            # Make sure all contexts have identical shapes.
            #
            # Our current Transformer requires this.
            # ---------------------------------------------

            context_shape = tuple(
                encoded_context.tensor.shape
            )

            if expected_context_shape is None:
                expected_context_shape = context_shape

            elif context_shape != expected_context_shape:
                raise ValueError(
                    "Problems have incompatible context sizes. "
                    f"Expected {expected_context_shape}, "
                    f"but problem {problem_index} produced "
                    f"{context_shape}."
                )

            # ---------------------------------------------
            # One example for every valid plan.
            # ---------------------------------------------

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