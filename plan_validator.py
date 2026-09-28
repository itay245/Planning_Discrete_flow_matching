from dataclasses import dataclass
from typing import Optional

import torch

from unified_planning.shortcuts import SequentialSimulator


@dataclass
class PlanValidationResult:
    """
    Result of validating a neural-network-generated plan.
    """

    valid: bool

    # True iff all actions were executable AND the final state
    # satisfies the planning goal.
    solves_problem: bool

    # Decoded symbolic action sequence.
    plan: list

    # Number of successfully executed actions.
    executed_actions: int

    # Index of the action/row where validation failed.
    # None if there was no execution/syntax failure.
    failure_step: Optional[int]

    # Human-readable explanation.
    reason: str

    # Final reachable state, when available.
    final_state: object = None

def _normalize_plan_tensor(
    plan_tensor: torch.Tensor,
    codec,
) -> torch.Tensor:
    """
    Normalize a generated plan tensor.

    Convention:
        H = maximum number of REAL actions
        number of neural plan slots = H + 1

    The additional slot is reserved for END.

    Expected structured shape:
        [H + 1, tuple_width]

    Expected flattened shape:
        [(H + 1) * tuple_width]
    """

    tensor = plan_tensor.detach().cpu()

    expected_rows = codec.horizon + 1
    expected_cols = codec.tuple_width

    # ---------------------------------------------
    # Flattened representation
    # ---------------------------------------------
    if tensor.ndim == 1:

        expected_size = (
            expected_rows
            * expected_cols
        )

        if tensor.numel() != expected_size:
            raise ValueError(
                f"Wrong flattened tensor size. "
                f"Horizon H={codec.horizon} allows "
                f"{codec.horizon} actions plus one END slot, "
                f"so expected {expected_size} categorical "
                f"entries, but got {tensor.numel()}."
            )

        tensor = tensor.reshape(
            expected_rows,
            expected_cols,
        )

    # ---------------------------------------------
    # Structured representation
    # ---------------------------------------------
    elif tensor.ndim == 2:

        expected_shape = (
            expected_rows,
            expected_cols,
        )

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Wrong tensor shape. "
                f"Horizon H={codec.horizon} means "
                f"H+1={expected_rows} plan slots "
                f"(actions + END). "
                f"Expected {expected_shape}, "
                f"got {tuple(tensor.shape)}."
            )

    else:
        raise ValueError(
            "Plan tensor must be either 1D "
            "(flattened) or 2D."
        )

    return tensor

def validate_tensor_plan(
    problem,
    plan_tensor: torch.Tensor,
    codec,
    strict_format: bool = True,
) -> PlanValidationResult:
    """
    Validate a generated plan tensor against a Unified Planning problem.

    Checks:

        1. Tensor shape.
        2. No MASK tokens remain.
        3. END / PAD structure is valid.
        4. Action schema exists.
        5. Correct number of arguments.
        6. Referenced objects exist.
        7. Every action is applicable when executed.
        8. Final state satisfies the goal.

    Parameters
    ----------
    problem:
        Unified Planning Problem.

    plan_tensor:
        Tensor containing categorical token IDs.

        Expected structured shape:

            [H + 1, tuple_width]

        A flattened tensor is also accepted.

    codec:
        The PlanCodec used to encode/decode the plan.

    strict_format:
        If True, enforce the canonical representation:

            ACTION
            ACTION
            ...
            END
            PAD
            PAD
            ...

        This is recommended for evaluating DFM outputs.

    Returns
    -------
    PlanValidationResult
    """

    # ---------------------------------------------------------
    # 1. Normalize tensor shape
    # ---------------------------------------------------------

    try:
        tensor = _normalize_plan_tensor(
        plan_tensor,
        codec,
    )   

    except ValueError as e:
        return PlanValidationResult(
            valid=False,
            solves_problem=False,
            plan=[],
            executed_actions=0,
            failure_step=None,
            reason=str(e),
        )

    # ---------------------------------------------------------
    # 2. Decode tensor rows into tokens
    # ---------------------------------------------------------

    rows = []

    try:
        for row in tensor:
            decoded_row = [
                codec.decode_token(token.item())
                for token in row
            ]
            rows.append(decoded_row)

    except ValueError as e:
        return PlanValidationResult(
            valid=False,
            solves_problem=False,
            plan=[],
            executed_actions=0,
            failure_step=None,
            reason=f"Unknown categorical token: {e}",
        )

    # ---------------------------------------------------------
    # Useful lookup tables
    # ---------------------------------------------------------

    actions_by_name = {
        action.name: action
        for action in problem.actions
    }

    objects_by_name = {
        obj.name: obj
        for obj in problem.all_objects
    }

    # ---------------------------------------------------------
    # 3. Parse the generated plan
    # ---------------------------------------------------------

    decoded_plan = []

    seen_end = False

    for row_index, row in enumerate(rows):

        operator = row[0]
        arguments = row[1:]

        # -----------------------------------------------------
        # MASK is never allowed in a completed generation.
        # -----------------------------------------------------

        if "<MASK>" in row:
            return PlanValidationResult(
                valid=False,
                solves_problem=False,
                plan=decoded_plan,
                executed_actions=0,
                failure_step=row_index,
                reason=(
                    f"MASK token remains in row "
                    f"{row_index}: {row}"
                ),
            )

        # -----------------------------------------------------
        # END
        # -----------------------------------------------------

        if operator == "<END>":

            if strict_format:
                expected_arguments = [
                    "<NONE>"
                ] * codec.max_action_arity

                if arguments != expected_arguments:
                    return PlanValidationResult(
                        valid=False,
                        solves_problem=False,
                        plan=decoded_plan,
                        executed_actions=0,
                        failure_step=row_index,
                        reason=(
                            "Malformed END row: "
                            f"{row}"
                        ),
                    )

            seen_end = True
            continue

        # -----------------------------------------------------
        # PAD
        # -----------------------------------------------------

        if operator == "<PAD>":

            if strict_format:

                if not seen_end:
                    return PlanValidationResult(
                        valid=False,
                        solves_problem=False,
                        plan=decoded_plan,
                        executed_actions=0,
                        failure_step=row_index,
                        reason=(
                            "PAD encountered before END "
                            f"at row {row_index}."
                        ),
                    )

                if any(
                    token != "<PAD>"
                    for token in row
                ):
                    return PlanValidationResult(
                        valid=False,
                        solves_problem=False,
                        plan=decoded_plan,
                        executed_actions=0,
                        failure_step=row_index,
                        reason=(
                            "Malformed PAD row: "
                            f"{row}"
                        ),
                    )

            continue

        # -----------------------------------------------------
        # No action may occur after END.
        # -----------------------------------------------------

        if seen_end:

            return PlanValidationResult(
                valid=False,
                solves_problem=False,
                plan=decoded_plan,
                executed_actions=0,
                failure_step=row_index,
                reason=(
                    "Action encountered after END: "
                    f"{row}"
                ),
            )

        # -----------------------------------------------------
        # Operator must correspond to a real action schema.
        # -----------------------------------------------------

        if operator not in actions_by_name:

            return PlanValidationResult(
                valid=False,
                solves_problem=False,
                plan=decoded_plan,
                executed_actions=0,
                failure_step=row_index,
                reason=(
                    f"Unknown action schema "
                    f"{operator!r}."
                ),
            )

        action = actions_by_name[operator]
        arity = len(action.parameters)

        # -----------------------------------------------------
        # Required arguments
        # -----------------------------------------------------

        real_arguments = arguments[:arity]

        for arg_index, argument in enumerate(
            real_arguments
        ):
            if argument in (
                "<NONE>",
                "<PAD>",
                "<END>",
                "<MASK>",
            ):
                return PlanValidationResult(
                    valid=False,
                    solves_problem=False,
                    plan=decoded_plan,
                    executed_actions=0,
                    failure_step=row_index,
                    reason=(
                        f"Missing/invalid argument "
                        f"{arg_index} of action "
                        f"{operator}: {row}"
                    ),
                )

            if argument not in objects_by_name:
                return PlanValidationResult(
                    valid=False,
                    solves_problem=False,
                    plan=decoded_plan,
                    executed_actions=0,
                    failure_step=row_index,
                    reason=(
                        f"Unknown object "
                        f"{argument!r} in action "
                        f"{operator}."
                    ),
                )

        # -----------------------------------------------------
        # Unused argument positions must be NONE.
        #
        # pickup(A) should be
        #
        #     [pickup, A, NONE]
        #
        # not
        #
        #     [pickup, A, B]
        # -----------------------------------------------------

        unused_arguments = arguments[arity:]

        if strict_format and any(
            arg != "<NONE>"
            for arg in unused_arguments
        ):
            return PlanValidationResult(
                valid=False,
                solves_problem=False,
                plan=decoded_plan,
                executed_actions=0,
                failure_step=row_index,
                reason=(
                    f"Action {operator} has unexpected "
                    f"arguments: {row}"
                ),
            )

        decoded_plan.append(
            [operator] + real_arguments
        )

    # ---------------------------------------------------------
    # In strict mode, require an explicit END.
    # ---------------------------------------------------------

    if strict_format and not seen_end:

        return PlanValidationResult(
            valid=False,
            solves_problem=False,
            plan=decoded_plan,
            executed_actions=0,
            failure_step=None,
            reason="Generated plan contains no END token.",
        )

    # ---------------------------------------------------------
    # 4. Execute the decoded plan using UP
    # ---------------------------------------------------------

    with SequentialSimulator(
        problem=problem
    ) as simulator:

        state = simulator.get_initial_state()

        # Empty plan may itself solve the problem.
        if len(decoded_plan) == 0:

            goal_reached = simulator.is_goal(state)

            return PlanValidationResult(
                valid=True,
                solves_problem=goal_reached,
                plan=[],
                executed_actions=0,
                failure_step=None,
                reason=(
                    "Empty plan satisfies the goal."
                    if goal_reached
                    else
                    "Empty plan is executable but "
                    "does not satisfy the goal."
                ),
                final_state=state,
            )

        # -----------------------------------------------------
        # Sequential execution
        # -----------------------------------------------------

        for step, symbolic_action in enumerate(
            decoded_plan
        ):

            action_name = symbolic_action[0]
            argument_names = symbolic_action[1:]

            action = actions_by_name[action_name]

            objects = [
                objects_by_name[name]
                for name in argument_names
            ]

            # Calling a UP action with objects constructs an
            # ActionInstance:
            #
            #     stack(A, B)
            #
            try:
                action_instance = action(*objects)

            except Exception as e:
                return PlanValidationResult(
                    valid=False,
                    solves_problem=False,
                    plan=decoded_plan,
                    executed_actions=step,
                    failure_step=step,
                    reason=(
                        f"Could not instantiate "
                        f"{symbolic_action}: {e}"
                    ),
                    final_state=state,
                )

            # -------------------------------------------------
            # Check applicability BEFORE applying.
            # -------------------------------------------------

            try:
                applicable = simulator.is_applicable(
                    state,
                    action_instance,
                )

            except Exception as e:
                return PlanValidationResult(
                    valid=False,
                    solves_problem=False,
                    plan=decoded_plan,
                    executed_actions=step,
                    failure_step=step,
                    reason=(
                        f"Error checking applicability of "
                        f"{symbolic_action}: {e}"
                    ),
                    final_state=state,
                )

            if not applicable:

                return PlanValidationResult(
                    valid=False,
                    solves_problem=False,
                    plan=decoded_plan,
                    executed_actions=step,
                    failure_step=step,
                    reason=(
                        f"Action {symbolic_action} is "
                        f"not applicable at step {step}."
                    ),
                    final_state=state,
                )

            # -------------------------------------------------
            # Apply transition
            # -------------------------------------------------

            next_state = simulator.apply(
                state,
                action_instance,
            )

            if next_state is None:

                return PlanValidationResult(
                    valid=False,
                    solves_problem=False,
                    plan=decoded_plan,
                    executed_actions=step,
                    failure_step=step,
                    reason=(
                        f"Failed to apply action "
                        f"{symbolic_action} at step {step}."
                    ),
                    final_state=state,
                )

            state = next_state

        # -----------------------------------------------------
        # 5. Goal test
        # -----------------------------------------------------

        goal_reached = simulator.is_goal(state)

        if goal_reached:

            return PlanValidationResult(
                valid=True,
                solves_problem=True,
                plan=decoded_plan,
                executed_actions=len(decoded_plan),
                failure_step=None,
                reason="Plan is valid and satisfies the goal.",
                final_state=state,
            )

        return PlanValidationResult(
            valid=True,
            solves_problem=False,
            plan=decoded_plan,
            executed_actions=len(decoded_plan),
            failure_step=None,
            reason=(
                "All actions are executable, but the "
                "final state does not satisfy the goal."
            ),
            final_state=state,
        )