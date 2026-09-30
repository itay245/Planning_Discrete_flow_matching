# Three-block DFM planning

Open **DFM_experiment.ipynb** and run its cells in order using the project's
Python environment. It imports the experiment functions from **main.py**;
it does not depend on running the older `Main.ipynb` first.

From this directory, the same experiment can be run with:

```bash
.venv/bin/python main.py
```

## Experiment

- Nine training problems and nine held-out problems, each with blocks A, B, C.
- Fixed seed 42. Initial arrangements are disjoint between splits; goals and
  object/action vocabulary may overlap. Stacks are listed bottom to top.
- Training targets are goal-satisfying trajectories of at most eight actions,
  enumerated by `trajectory_generator.py` and checked by `plan_validator.py`.
  At most 128 plans per problem are retained. The default split yields 320 plans.
- One shared `PlanCodec` and `PlanningDFMDataset`; only training problems
  contribute examples. Problems with more plans contribute more training weight.
- A two-layer, 64-dimensional conditional Transformer trains for 50 epochs
  using the existing DFM training loss and sampling implementation.
- Each held-out problem gets 32 samples with 64 flow steps. Every raw sample
  is evaluated, including malformed and non-executable plans. No plan repair
  or reference solution is substituted for a generated sample.

**Satisfaction rate** is satisfying plans divided by all generated plans.
**Executable rate** includes well-formed, executable plans that miss the goal.
**Problems solved** counts problems with at least one satisfying sample, so
it depends on the sampling budget. These are distinct from training token accuracy.

## Recorded run

The executed notebook records the default 50-epoch CPU run: **0/288 satisfying
plans and 0/9 problems solved**, with a 0% executable rate. Example failures
include malformed arguments and PAD before END. Tests independently establish
that all held-out problems have solutions within the horizon; zero model
successes do not mean the problems are unsatisfiable. This is a small baseline
experiment, not evidence of successful generalization.

## Artifacts and configuration

`artifacts/three_blocks/` contains:

- `dataset.pt`: encoded training examples, vocabulary, and horizon.
- `model.pt`: trained parameters, model/experiment configuration, vocabulary,
  and training history.
- `results.json`: exact split, plan counts, loss history, per-problem metrics,
  and every generated token tensor with its validation result.

Artifacts are ignored by Git; the executed notebook retains readable results.
A repeat run overwrites the chosen output directory. To keep a separate run:

```bash
.venv/bin/python main.py --epochs 100 --output-dir artifacts/longer_run
```

Other settings are fields of `ExperimentConfig` in the notebook. The CLI also
accepts `--samples-per-problem` and `--sampling-steps`. Keep evaluation settings
fixed when comparing experiments and use a separate development split if tuning
hyperparameters; do not tune against the final held-out results.

## Checks

```bash
.venv/bin/python -m unittest test_experiment problem_generator -v
```

The tests cover problem construction, split separation, valid training targets,
held-out feasibility, metric denominators, and strict rejection of duplicate END
markers. The nine existing generator tests and four experiment tests pass.
