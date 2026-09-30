"""Integration checks: python -m unittest test_experiment -v."""
import unittest
from unittest.mock import patch

import torch

from main import (ExperimentConfig, arrangement_facts, block_arrangements,
                  build_training_dataset, evaluate_model, make_problem_splits)
from converter import PlanCodec
from plan_validator import validate_tensor_plan
from trajectory_generator import enumerate_trajectories_up_to_horizon


class ExperimentTests(unittest.TestCase):
    def test_splits_are_reproducible_and_disjoint(self):
        train, test = make_problem_splits()
        self.assertEqual(len(block_arrangements()), 13)
        self.assertEqual((len(train), len(test)), (9, 9))
        signature = lambda item: (item['initial_stacks'], item['goal_stacks'])
        self.assertEqual(len({signature(item) for item in train + test}), 18)
        self.assertFalse({item['initial_stacks'] for item in train} &
                         {item['initial_stacks'] for item in test})
        again = make_problem_splits()
        self.assertEqual([signature(item) for item in train + test],
                         [signature(item) for split in again for item in split])
        for item in train + test:
            self.assertEqual({o.name for o in item['problem'].all_objects}, {'A', 'B', 'C'})
            self.assertNotEqual(item['initial_stacks'], item['goal_stacks'])
            self.assertEqual(len([f for f in arrangement_facts(item['initial_stacks'])
                                  if f[0] in ('on', 'ontable')]), 3)

    def test_dataset_and_held_out_feasibility(self):
        train, test = make_problem_splits()
        config = ExperimentConfig(max_plans_per_problem=3)
        dataset, codec, counts = build_training_dataset(train, config)
        self.assertEqual(len(counts), 9)
        self.assertEqual(len(dataset), sum(row['training_plans'] for row in counts))
        train_contexts = {tuple(example['context'].flatten().tolist()) for example in dataset.examples}
        self.assertEqual(len(train_contexts), 9)
        for item in test:
            problem = item['problem']
            context = codec.encode_problem_context(problem)
            self.assertNotIn(tuple(context.tensor.flatten().tolist()), train_contexts)
            # Test-only witnesses: never used as training data or model predictions.
            plans = enumerate_trajectories_up_to_horizon(problem, config.horizon, k=0)['plans']
            self.assertTrue(plans, item['id'])
            self.assertTrue(validate_tensor_plan(problem, codec.encode_plan(plans[0]).tensor, codec).solves_problem)

    def test_metrics_count_invalid_and_unsatisfying_samples(self):
        instance = make_problem_splits()[1][0]
        problem = instance['problem']
        codec = PlanCodec.from_problem(problem, horizon=8)
        plans = enumerate_trajectories_up_to_horizon(problem, 8, k=0)['plans']
        valid = codec.encode_plan(plans[0]).tensor
        empty = codec.encode_plan([]).tensor
        invalid = torch.full_like(valid, codec.mask_id)
        with patch('main.sample_dfm', return_value=torch.stack([valid, empty, invalid])):
            report = evaluate_model(None, [instance], codec, ExperimentConfig(samples_per_problem=3))
        self.assertEqual(report['summary']['total_samples'], 3)
        self.assertEqual(report['summary']['satisfaction_rate'], 1 / 3)
        self.assertEqual(report['summary']['executable_rate'], 2 / 3)
        self.assertEqual(report['summary']['solved_problems'], 1)

    def test_strict_validator_rejects_duplicate_end(self):
        problem = make_problem_splits()[0][0]['problem']
        codec = PlanCodec.from_problem(problem, horizon=8)
        tensor = codec.encode_plan([]).tensor
        tensor[1] = tensor[0]
        result = validate_tensor_plan(problem, tensor, codec)
        self.assertFalse(result.valid)
        self.assertIn('Duplicate END', result.reason)


if __name__ == '__main__':
    unittest.main()
