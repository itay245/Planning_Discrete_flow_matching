"""Reproducible three-block DFM experiment.

Run ``python main.py`` or use the cells in ``DFM_experiment.ipynb``.
Training uses nine problems; evaluation uses nine new problems whose initial
arrangements never occur as training contexts. Both splits share A, B, C and
the same action vocabulary. Held-out problems never contribute target plans.
"""
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
import argparse
import json
import random

import torch
from torch.utils.data import DataLoader

from converter import PlanCodec
from dataset import PlanningDFMDataset
from flow import sample_dfm
from model import DFMConfig, PlanningDFM, build_plan_output_mask
from plan_validator import validate_tensor_plan
from problem_generator import create_blocksworld_problem
from train import train_dfm
from trajectory_generator import enumerate_trajectories_up_to_horizon


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    horizon: int = 8
    max_plans_per_problem: int = 1000
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    samples_per_problem: int = 32
    sampling_steps: int = 64
    d_model: int = 64
    num_layers: int = 2
    cpu_threads: int = 4
    output_dir: str = "artifacts/three_blocks"

    def __post_init__(self):
        for name in ("horizon", "max_plans_per_problem", "epochs", "batch_size",
                     "samples_per_problem", "sampling_steps", "d_model", "num_layers", "cpu_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % 4:
            raise ValueError("d_model must be divisible by four attention heads")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def block_arrangements():
    """All 13 table arrangements; each stack is listed bottom to top."""
    arrangements = set()
    for order in permutations(("A", "B", "C")):
        for cuts in range(4):
            stacks = [[order[0]]]
            for index, block in enumerate(order[1:]):
                if cuts & (1 << index):
                    stacks.append([])
                stacks[-1].append(block)
            arrangements.add(tuple(sorted(tuple(stack) for stack in stacks)))
    return sorted(arrangements)


def arrangement_facts(stacks):
    """Convert a legal table arrangement into its true STRIPS facts."""
    facts = {("handempty",): True}
    for stack in stacks:
        facts[("ontable", stack[0])] = True
        facts[("clear", stack[-1])] = True
        for lower, upper in zip(stack, stack[1:]):
            facts[("on", upper, lower)] = True
    return facts


def make_problem_splits(seed=42,num_probs=9):
    """Select 9 distinct pairs per split, with disjoint initial arrangements.

    Goals describe complete table arrangements using positive facts, as required
    by PlanCodec. No instance is already solved in its initial state. Goals may
    be shared across splits; the initial-state/goal pairs and initial states are
    disjoint. This is a small same-domain generalization experiment.
    """
    rng = random.Random(seed)
    arrangements = block_arrangements()
    rng.shuffle(arrangements)
    splits = []
    for split, starts in (("train", arrangements[:6]), ("test", arrangements[6:])):
        # Cycle over starts so every available initial arrangement is represented.
        pairs = []
        for start in starts:
            goals = [goal for goal in arrangements if goal != start]
            rng.shuffle(goals)
            pairs.append([(start, goal) for goal in goals])
        selected = [pairs[i % len(starts)][i // len(starts)] for i in range(num_probs)]
        instances = []
        for index, (start, goal) in enumerate(selected, 1):
            config = {"blocks": ["A", "B", "C"],
                      "initial_state": arrangement_facts(start),
                      "goal_state": arrangement_facts(goal)}
            instances.append({"id": f"{split}_{index:02d}", "initial_stacks": start,
                              "goal_stacks": goal, "problem": create_blocksworld_problem(config)})
        splits.append(instances)
    return tuple(splits)


def build_training_dataset(instances, config):
    """Enumerate valid plans, verify targets, and encode with one shared codec."""
    codec = PlanCodec.from_problem(instances[0]["problem"], horizon=config.horizon)
    pairs, counts = [], []
    rng = random.Random(config.seed)
    for instance in instances:
        problem = instance["problem"]
        result = enumerate_trajectories_up_to_horizon(
            problem, horizon=config.horizon, k=0, seed=config.seed)
        plans = result["plans"]
        if not plans:
            raise ValueError(f"{instance['id']} has no plan within horizon {config.horizon}")
        if len(plans) > config.max_plans_per_problem:
            plans = rng.sample(plans, config.max_plans_per_problem)
        for plan in plans:
            validation = validate_tensor_plan(problem, codec.encode_plan(plan).tensor, codec)
            if not validation.solves_problem:
                raise RuntimeError(f"Invalid training target: {validation.reason}")
        pairs.append((problem, plans))
        counts.append({"id": instance["id"], "enumerated_plans": result["num_plans"],
                       "training_plans": len(plans)})
    return PlanningDFMDataset.from_problems(pairs, codec), codec, counts


def train_model(dataset, codec, config, device=None):
    """Train only on the supplied training dataset; return model and loss history."""
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    model_config = DFMConfig(
        vocab_size=codec.vocab_size, tuple_width=codec.tuple_width,
        context_rows=dataset[0]["context"].shape[0], plan_slots=codec.num_plan_slots,
        d_model=config.d_model, nhead=4, num_layers=config.num_layers,
        dim_feedforward=16 * config.d_model, dropout=0.1)
    model = PlanningDFM(model_config, build_plan_output_mask(codec))
    history = train_dfm(model, loader, codec, epochs=config.epochs,
                        learning_rate=config.learning_rate, device=device)
    return model, history


def evaluate_model(model, instances, codec, config):
    """Validate every raw sample, including malformed and non-executable outputs.

    satisfaction_rate = goal-satisfying samples / all generated samples.
    solved_problem_rate = problems with at least one satisfying sample / 9.
    No plan repair, filtering, or reference-plan substitution is performed.
    """
    torch.manual_seed(config.seed + 1)
    rows, samples = [], []
    for instance in instances:
        problem = instance["problem"]
        context = codec.encode_problem_context(problem)
        generated = sample_dfm(
            model, context.tensor.unsqueeze(0).repeat(config.samples_per_problem, 1, 1),
            context.attention_mask.unsqueeze(0).repeat(config.samples_per_problem, 1),
            codec, num_steps=config.sampling_steps).cpu()
        validations = [validate_tensor_plan(problem, tensor, codec, strict_format=True)
                       for tensor in generated]
        satisfied = sum(result.solves_problem for result in validations)
        executable = sum(result.valid for result in validations)
        rows.append({"id": instance["id"], "samples": len(validations),
                     "executable": executable, "satisfied": satisfied,
                     "satisfaction_rate": satisfied / len(validations), "solved": satisfied > 0})
        for index, (tensor, result) in enumerate(zip(generated, validations)):
            samples.append({"id": instance["id"], "sample": index, "tokens": tensor.tolist(),
                            "plan": result.plan, "valid": result.valid,
                            "solves_problem": result.solves_problem, "reason": result.reason,
                            "failure_step": result.failure_step})
    total = sum(row["samples"] for row in rows)
    satisfied = sum(row["satisfied"] for row in rows)
    solved = sum(row["solved"] for row in rows)
    return {"per_problem": rows, "samples": samples, "summary": {
        "total_samples": total, "satisfying_samples": satisfied,
        "satisfaction_rate": satisfied / total,
        "executable_rate": sum(row["executable"] for row in rows) / total,
        "solved_problems": solved, "total_problems": len(rows),
        "solved_problem_rate": solved / len(rows)}}


def save_experiment(config, train_instances, test_instances, dataset, codec, counts, model, history, evaluation):
    """Persist the exact split, encoded dataset, checkpoint, and raw validation results."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    def instance_manifest(instances):
        return [
            {key: value for key, value in item.items() if key != "problem"}
            for item in instances
        ]

    report = {"config": asdict(config), "train_instances": instance_manifest(train_instances),
              "test_instances": instance_manifest(test_instances), "dataset_counts": counts,
              "history": history, "evaluation": evaluation}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"examples": dataset.examples, "token_to_id": codec.token_to_id,
                "horizon": codec.horizon}, output / "dataset.pt")
    torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(model.config),
                "experiment_config": asdict(config), "token_to_id": codec.token_to_id,
                "history": history}, output / "model.pt")


def print_evaluation(evaluation):
    print("\nHeld-out evaluation (raw samples, strict plan validation)")
    print(f"{'Problem':<12} {'Executable':>12} {'Satisfying':>12} {'Satisfaction':>14}")
    for row in evaluation["per_problem"]:
        print(f"{row['id']:<12} {row['executable']:>6}/{row['samples']:<5} "
              f"{row['satisfied']:>6}/{row['samples']:<5} {row['satisfaction_rate']:>13.1%}")
    summary = evaluation["summary"]
    print(f"Overall satisfaction: {summary['satisfaction_rate']:.1%}; "
          f"problems solved: {summary['solved_problems']}/{summary['total_problems']}")


def run_experiment(config=None):
    config = config or ExperimentConfig()
    train_instances, test_instances = make_problem_splits(config.seed)
    dataset, codec, counts = build_training_dataset(train_instances, config)
    print(f"Training: {len(dataset)} valid plans from 9 problems; held out: 9 problems.")
    model, history = train_model(dataset, codec, config)
    evaluation = evaluate_model(model, test_instances, codec, config)
    save_experiment(config, train_instances, test_instances, dataset, codec, counts,
                    model, history, evaluation)
    print_evaluation(evaluation)
    print(f"Saved dataset, model, and report to {config.output_dir}")
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-problem", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts/three_blocks")
    args = parser.parse_args()
    run_experiment(ExperimentConfig(**vars(args)))
