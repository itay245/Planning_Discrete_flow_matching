import random

from unified_planning.shortcuts import Compiler, SequentialSimulator
from unified_planning.engines import CompilationKind
from unified_planning.plans import ActionInstance


def _parameter_name(parameter):
    if parameter.is_object_exp():
        return parameter.object().name
    return str(parameter)


def _action_to_tuple(action_instance, max_arity):
    """
    stack(A, B) -> ["stack", "A", "B"]
    pick_up(A)  -> ["pick_up", "A", None]
    """
    result = [action_instance.action.name]

    result.extend(
        _parameter_name(p)
        for p in action_instance.actual_parameters
    )

    while len(result) < max_arity + 1:
        result.append(None)

    return result


def _pad_trajectory(trajectory, horizon, tuple_size):
    """
    If H is the maximum NUMBER OF ACTIONS, we use H+1 positions:

        actions..., END, PAD, PAD, ...

    Example, H=4:

        [pickup(A), stack(A,B)]

    becomes

        [
            pickup(A),
            stack(A,B),
            END,
            PAD,
            PAD
        ]

    Total sequence length = H + 1.
    """

    result = [list(action) for action in trajectory]

    end_token = ["END"] + [None] * (tuple_size - 1)
    pad_token = ["PAD"] + [None] * (tuple_size - 1)

    result.append(end_token)

    while len(result) < horizon + 1:
        result.append(list(pad_token))

    return result


def enumerate_trajectories_up_to_horizon(
    problem,
    horizon,
    k,
    seed=None,
    pad=True,
    include_empty_plan=False,
):
    """
    Enumerate ALL executable trajectories with lengths 1..H
    in a single DFS traversal.

    Returns:
        - every trajectory of length <= H whose final state
          satisfies the goal;
        - a uniform random sample of at most k of all other
          executable trajectories of length <= H.

    Therefore:

        len(result["trajectories"])
        =
        min(k, number of non-goal trajectories)

    A trajectory that reaches the goal at depth d is recorded
    as a plan, but search continues below it. Therefore longer
    executable sequences may also be plans if their final states
    satisfy the goal.
    """

    if horizon < 0:
        raise ValueError("horizon must be >= 0")

    if k < 0:
        raise ValueError("k must be >= 0")

    rng = random.Random(seed)

    max_arity = max(
        (len(action.parameters) for action in problem.actions),
        default=0,
    )

    tuple_size = max_arity + 1

    plans = []

    # Uniform reservoir sample of non-plan trajectories.
    sampled_trajectories = []
    num_non_plan_trajectories = 0

    # Useful statistics.
    num_executable_trajectories = 0

    # Optional counts by trajectory length.
    plans_by_length = {
        length: 0
        for length in range(horizon + 1)
    }

    trajectories_by_length = {
        length: 0
        for length in range(horizon + 1)
    }

    def store_plan(trajectory):
        if pad:
            plans.append(
                _pad_trajectory(
                    trajectory,
                    horizon,
                    tuple_size,
                )
            )
        else:
            plans.append(list(trajectory))

    def store_non_plan(trajectory):
        nonlocal num_non_plan_trajectories

        num_non_plan_trajectories += 1

        if k == 0:
            return

        if pad:
            candidate = _pad_trajectory(
                trajectory,
                horizon,
                tuple_size,
            )
        else:
            candidate = list(trajectory)

        # Initially fill the reservoir.
        if len(sampled_trajectories) < k:
            sampled_trajectories.append(candidate)
            return

        # Standard reservoir sampling.
        #
        # Every non-plan trajectory encountered so far has
        # equal probability of appearing in the final sample.
        j = rng.randrange(num_non_plan_trajectories)

        if j < k:
            sampled_trajectories[j] = candidate

    # ---------------------------------------------------------
    # Ground the problem once.
    # ---------------------------------------------------------

    with Compiler(
        problem_kind=problem.kind,
        compilation_kind=CompilationKind.GROUNDING,
    ) as grounder:

        grounding_result = grounder.compile(
            problem,
            CompilationKind.GROUNDING,
        )

        grounded_problem = grounding_result.problem
        grounded_actions = list(grounded_problem.actions)

        with SequentialSimulator(
            problem=grounded_problem
        ) as simulator:

            initial_state = simulator.get_initial_state()

            # -------------------------------------------------
            # Optional zero-length plan.
            # -------------------------------------------------

            if (
                include_empty_plan
                and simulator.is_goal(initial_state)
            ):
                store_plan([])
                plans_by_length[0] += 1

            # -------------------------------------------------
            # One DFS explores ALL horizons 1...H.
            # -------------------------------------------------

            def dfs(state, trajectory):
                nonlocal num_executable_trajectories

                depth = len(trajectory)

                # We have already classified this prefix.
                # Do not expand beyond H.
                if depth == horizon:
                    return

                for grounded_action in grounded_actions:

                    grounded_ai = ActionInstance(
                        grounded_action
                    )

                    if not simulator.is_applicable(
                        state,
                        grounded_ai,
                    ):
                        continue

                    next_state = simulator.apply(
                        state,
                        grounded_ai,
                    )

                    if next_state is None:
                        continue

                    original_ai = (
                        grounding_result
                        .map_back_action_instance(
                            grounded_ai
                        )
                    )

                    encoded_action = _action_to_tuple(
                        original_ai,
                        max_arity,
                    )

                    # Add one action to the current prefix.
                    trajectory.append(encoded_action)

                    new_depth = len(trajectory)

                    # -----------------------------------------
                    # THIS is the important change:
                    #
                    # classify every prefix, not only depth H.
                    # -----------------------------------------

                    num_executable_trajectories += 1

                    if simulator.is_goal(next_state):

                        store_plan(trajectory)

                        plans_by_length[new_depth] += 1

                    else:

                        store_non_plan(trajectory)

                        trajectories_by_length[new_depth] += 1

                    # -----------------------------------------
                    # Continue searching until H regardless of
                    # whether this prefix already reaches goal.
                    # -----------------------------------------

                    dfs(
                        next_state,
                        trajectory,
                    )

                    trajectory.pop()

            dfs(
                initial_state,
                [],
            )

    return {
        "plans": plans,
        "trajectories": sampled_trajectories,

        "num_plans": len(plans),

        "num_non_plan_trajectories":
            num_non_plan_trajectories,

        "num_executable_trajectories":
            num_executable_trajectories,

        "plans_by_length":
            plans_by_length,

        "trajectories_by_length":
            trajectories_by_length,

        "horizon": horizon,
    }