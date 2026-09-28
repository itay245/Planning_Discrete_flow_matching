from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


# ============================================================
# Special tokens
# ============================================================

PAD = "<PAD>"
MASK = "<MASK>"
NONE = "<NONE>"
END = "<END>"

INIT = "<INIT>"
GOAL = "<GOAL>"

SPECIAL_TOKENS = [
    PAD,
    MASK,
    NONE,
    END,
    INIT,
    GOAL,
]


# ============================================================
# Returned data structures
# ============================================================

@dataclass
class EncodedPlan:
    """
    Neural representation of one plan.

    tensor:
        Shape:
            [H + 1, tuple_width]

        H is the maximum number of REAL actions.

        The additional row is required for END.

    length:
        Number of real actions in the plan.
        Does not include END or PAD.
    """

    tensor: torch.LongTensor
    length: int


@dataclass
class EncodedContext:
    """
    Neural representation of the initial state and goal.

    tensor:
        Shape:
            [context_rows, tuple_width]

    attention_mask:
        Shape:
            [context_rows]

        True:
            meaningful row

        False:
            padding row

    num_initial_facts:
        Number of true facts in the initial state.

    num_goal_facts:
        Number of positive goal facts.
    """

    tensor: torch.LongTensor
    attention_mask: torch.BoolTensor

    num_initial_facts: int
    num_goal_facts: int


# ============================================================
# Plan / Planning Context Codec
# ============================================================

class PlanCodec:
    """
    Converts between symbolic classical-planning representations
    and categorical PyTorch tensors.

    The same vocabulary is used for:

        - action schemas
        - predicate schemas
        - objects
        - special tokens

    Example action:

        stack(A, B)

    becomes

        ["stack", "A", "B"]

    Example unary action:

        pick_up(A)

    becomes

        ["pick_up", "A", "<NONE>"]

    Example fluent:

        on(A, B)

    becomes

        ["on", "A", "B"]

    Example unary fluent:

        clear(A)

    becomes

        ["clear", "A", "<NONE>"]


    Horizon convention
    ------------------

    H = maximum number of REAL actions.

    Therefore:

        num_plan_slots = H + 1

    because every plan representation also requires an END row.

    For H = 4:

        [a1]
        [a2]
        [END]
        [PAD]
        [PAD]

    and a maximum-length plan is:

        [a1]
        [a2]
        [a3]
        [a4]
        [END]
    """

    # ========================================================
    # Construction
    # ========================================================

    def __init__(
        self,
        action_names: Sequence[str],
        predicate_names: Sequence[str],
        object_names: Sequence[str],
        max_action_arity: int,
        max_fluent_arity: int,
        horizon: int,
    ):

        if horizon < 0:
            raise ValueError(
                "horizon must be non-negative."
            )

        self.action_names = sorted(
            set(action_names)
        )

        self.predicate_names = sorted(
            set(predicate_names)
        )

        self.object_names = sorted(
            set(object_names)
        )

        self.max_action_arity = (
            max_action_arity
        )

        self.max_fluent_arity = (
            max_fluent_arity
        )

        # ----------------------------------------------------
        # One common tuple width is used everywhere.
        #
        # First entry:
        #     operator / predicate
        #
        # Remaining entries:
        #     arguments
        # ----------------------------------------------------

        self.max_arity = max(
            max_action_arity,
            max_fluent_arity,
        )

        self.tuple_width = (
            1 + self.max_arity
        )

        # ----------------------------------------------------
        # H = max REAL actions.
        #
        # Need an extra row for END.
        # ----------------------------------------------------

        self.horizon = horizon

        self.num_plan_slots = (
            horizon + 1
        )

        # ----------------------------------------------------
        # Construct one shared categorical vocabulary.
        # ----------------------------------------------------

        vocabulary = list(
            dict.fromkeys(
                SPECIAL_TOKENS
                + self.action_names
                + self.predicate_names
                + self.object_names
            )
        )

        self.token_to_id = {
            token: index
            for index, token
            in enumerate(vocabulary)
        }

        self.id_to_token = {
            index: token
            for token, index
            in self.token_to_id.items()
        }

        self.vocab_size = len(
            vocabulary
        )

        # ----------------------------------------------------
        # Convenient special-token IDs
        # ----------------------------------------------------

        self.pad_id = (
            self.token_to_id[PAD]
        )

        self.mask_id = (
            self.token_to_id[MASK]
        )

        self.none_id = (
            self.token_to_id[NONE]
        )

        self.end_id = (
            self.token_to_id[END]
        )

        self.init_id = (
            self.token_to_id[INIT]
        )

        self.goal_id = (
            self.token_to_id[GOAL]
        )

    # ========================================================
    # Construct directly from Unified Planning problem
    # ========================================================

    @classmethod
    def from_problem(
        cls,
        problem,
        horizon: int,
    ) -> "PlanCodec":

        action_names = [
            action.name
            for action in problem.actions
        ]

        predicate_names = [
            fluent.name
            for fluent in problem.fluents
        ]

        object_names = [
            obj.name
            for obj in problem.all_objects
        ]

        max_action_arity = max(
            (
                len(action.parameters)
                for action in problem.actions
            ),
            default=0,
        )

        max_fluent_arity = max(
            (
                len(fluent.signature)
                for fluent in problem.fluents
            ),
            default=0,
        )

        return cls(
            action_names=action_names,
            predicate_names=predicate_names,
            object_names=object_names,
            max_action_arity=max_action_arity,
            max_fluent_arity=max_fluent_arity,
            horizon=horizon,
        )

    # ========================================================
    # Basic categorical conversion
    # ========================================================

    def encode_token(
        self,
        token,
    ) -> int:
        """
        Convert a symbolic token to its categorical ID.

        Python None is interpreted as <NONE>.
        """

        if token is None:
            token = NONE

        token = str(token)

        if token not in self.token_to_id:
            raise ValueError(
                f"Unknown token {token!r}. "
                f"Known vocabulary: "
                f"{list(self.token_to_id.keys())}"
            )

        return self.token_to_id[token]

    def decode_token(
        self,
        token_id: int,
    ) -> str:
        """
        Convert categorical ID back into symbolic token.
        """

        token_id = int(token_id)

        if token_id not in self.id_to_token:
            raise ValueError(
                f"Unknown token ID: {token_id}"
            )

        return self.id_to_token[
            token_id
        ]

    # ========================================================
    # Standard special rows
    # ========================================================

    def end_row(self) -> list[str]:
        """
        [END, NONE, ..., NONE]
        """

        return (
            [END]
            + [NONE]
            * (self.tuple_width - 1)
        )

    def pad_row(self) -> list[str]:
        """
        [PAD, PAD, ..., PAD]
        """

        return (
            [PAD]
            * self.tuple_width
        )

    def mask_row(self) -> list[str]:
        """
        [MASK, MASK, ..., MASK]
        """

        return (
            [MASK]
            * self.tuple_width
        )

    def init_row(self) -> list[str]:
        """
        [INIT, NONE, ..., NONE]
        """

        return (
            [INIT]
            + [NONE]
            * (self.tuple_width - 1)
        )

    def goal_row(self) -> list[str]:
        """
        [GOAL, NONE, ..., NONE]
        """

        return (
            [GOAL]
            + [NONE]
            * (self.tuple_width - 1)
        )

    # ========================================================
    # Action representation
    # ========================================================

    def normalize_action(
        self,
        action: Sequence,
    ) -> list[str]:
        """
        Normalize an action to tuple_width entries.

        Examples:

            ["pick_up", "A"]
                ->
            ["pick_up", "A", "<NONE>"]

            ["pick_up", "A", None]
                ->
            ["pick_up", "A", "<NONE>"]

            ["stack", "A", "B"]
                ->
            ["stack", "A", "B"]
        """

        if len(action) == 0:
            raise ValueError(
                "Action cannot be empty."
            )

        if len(action) > self.tuple_width:
            raise ValueError(
                f"Action {action} contains "
                f"{len(action)} fields, but "
                f"tuple width is "
                f"{self.tuple_width}."
            )

        result = []

        for token in action:

            if token is None:
                result.append(NONE)

            else:
                result.append(
                    str(token)
                )

        while (
            len(result)
            < self.tuple_width
        ):
            result.append(NONE)

        return result

    def encode_action(
        self,
        action: Sequence,
    ) -> torch.LongTensor:
        """
        Encode one symbolic action tuple.
        """

        normalized = (
            self.normalize_action(action)
        )

        return torch.tensor(
            [
                self.encode_token(token)
                for token in normalized
            ],
            dtype=torch.long,
        )

    # ========================================================
    # Plan encoding
    # ========================================================

    def encode_plan(
        self,
        plan: Sequence[Sequence],
    ) -> EncodedPlan:
        """
        Encode a symbolic plan.

        Accepts either:

        1. Real actions only:

            [
                ["pick_up", "A", None],
                ["stack", "A", "B"],
            ]

        or

        2. A previously padded representation:

            [
                ["pick_up", "A", None],
                ["stack", "A", "B"],
                ["END", None, None],
                ["PAD", None, None],
                ...
            ]

        Both are converted to the canonical representation:

            actions...
            END
            PAD
            ...
            PAD

        with exactly H + 1 rows.
        """

        action_rows = []

        # ----------------------------------------------------
        # Extract only REAL actions from input.
        # ----------------------------------------------------

        for row_index, row in enumerate(
            plan
        ):

            if len(row) == 0:
                raise ValueError(
                    f"Empty row at "
                    f"index {row_index}."
                )

            operator = str(row[0])

            # Accept both older enumerator
            # representations and canonical tokens.
            if operator in (
                "END",
                END,
            ):
                break

            if operator in (
                "PAD",
                PAD,
            ):
                break

            action_rows.append(
                self.normalize_action(row)
            )

        # ----------------------------------------------------
        # Horizon applies ONLY to real actions.
        # ----------------------------------------------------

        action_length = len(
            action_rows
        )

        if (
            action_length
            > self.horizon
        ):
            raise ValueError(
                f"Plan contains "
                f"{action_length} real actions, "
                f"but horizon H="
                f"{self.horizon} allows at most "
                f"{self.horizon} actions."
            )

        # ----------------------------------------------------
        # Canonical form:
        #
        # action...
        # END
        # PAD...
        # ----------------------------------------------------

        rows = list(
            action_rows
        )

        rows.append(
            self.end_row()
        )

        while (
            len(rows)
            < self.num_plan_slots
        ):
            rows.append(
                self.pad_row()
            )

        if (
            len(rows)
            != self.num_plan_slots
        ):
            raise RuntimeError(
                "Internal codec error: "
                f"expected "
                f"{self.num_plan_slots} rows, "
                f"got {len(rows)}."
            )

        tensor = torch.tensor(
            [
                [
                    self.encode_token(
                        token
                    )
                    for token in row
                ]
                for row in rows
            ],
            dtype=torch.long,
        )

        return EncodedPlan(
            tensor=tensor,
            length=action_length,
        )

    # ========================================================
    # Plan decoding
    # ========================================================

    def decode_plan(
        self,
        tensor: torch.Tensor,
    ) -> list[list[str | None]]:
        """
        Decode a neural tensor into REAL ACTIONS only.

        END and PAD are representation-level tokens and are
        therefore not returned.

        Desired round-trip property:

            decode_plan(
                encode_plan(plan).tensor
            )

        returns the original real action sequence.
        """

        tensor = (
            tensor
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # Accept flattened tensors too.
        # ----------------------------------------------------

        if tensor.ndim == 1:

            expected_size = (
                self.num_plan_slots
                * self.tuple_width
            )

            if (
                tensor.numel()
                != expected_size
            ):
                raise ValueError(
                    f"Expected "
                    f"{expected_size} categorical "
                    f"entries, got "
                    f"{tensor.numel()}."
                )

            tensor = tensor.reshape(
                self.num_plan_slots,
                self.tuple_width,
            )

        expected_shape = (
            self.num_plan_slots,
            self.tuple_width,
        )

        if (
            tuple(tensor.shape)
            != expected_shape
        ):
            raise ValueError(
                f"Expected tensor shape "
                f"{expected_shape}, got "
                f"{tuple(tensor.shape)}."
            )

        plan = []

        for row_index, row in enumerate(
            tensor
        ):

            tokens = [
                self.decode_token(
                    token.item()
                )
                for token in row
            ]

            operator = tokens[0]

            # ------------------------------------------------
            # END terminates real plan.
            # ------------------------------------------------

            if operator == END:
                break

            # ------------------------------------------------
            # PAD before END is malformed, but this function
            # is a decoder rather than validator.
            # ------------------------------------------------

            if operator == PAD:
                break

            if operator == MASK:
                raise ValueError(
                    "Cannot decode completed "
                    f"plan: MASK remains "
                    f"at row {row_index}."
                )

            action = [
                operator
            ]

            for token in tokens[1:]:

                if token == NONE:
                    action.append(None)

                elif token in (
                    PAD,
                    END,
                    MASK,
                    INIT,
                    GOAL,
                ):
                    raise ValueError(
                        f"Invalid token {token} "
                        f"inside action at row "
                        f"{row_index}: {tokens}"
                    )

                else:
                    action.append(
                        token
                    )

            plan.append(
                action
            )

        return plan

    # ========================================================
    # DFM all-mask source plan
    # ========================================================

    def masked_plan(
        self,
    ) -> torch.LongTensor:
        """
        Return the DFM source plan X_0.

        Shape:

            [H + 1, tuple_width]

        Every categorical variable initially contains MASK.
        """

        return torch.full(
            (
                self.num_plan_slots,
                self.tuple_width,
            ),
            fill_value=self.mask_id,
            dtype=torch.long,
        )

    # ========================================================
    # Fluent representation
    # ========================================================

    def fluent_to_tuple(
        self,
        fluent_exp,
    ) -> list[str]:
        """
        Convert a grounded Unified Planning fluent expression
        to the shared symbolic tuple representation.

        Examples:

            on(A, B)
                ->
            ["on", "A", "B"]

            clear(A)
                ->
            ["clear", "A", "<NONE>"]

            handempty
                ->
            ["handempty", "<NONE>", "<NONE>"]
        """

        if (
            not fluent_exp
            .is_fluent_exp()
        ):
            raise ValueError(
                "Expected grounded fluent "
                f"expression, got "
                f"{fluent_exp}."
            )

        result = [
            fluent_exp
            .fluent()
            .name
        ]

        for arg in fluent_exp.args:

            if not arg.is_object_exp():
                raise ValueError(
                    "Only grounded object "
                    "arguments are currently "
                    "supported. "
                    f"Got {arg} in "
                    f"{fluent_exp}."
                )

            result.append(
                arg.object().name
            )

        while (
            len(result)
            < self.tuple_width
        ):
            result.append(NONE)

        if (
            len(result)
            > self.tuple_width
        ):
            raise ValueError(
                f"Fluent {fluent_exp} "
                "exceeds codec tuple width."
            )

        return result

    # ========================================================
    # Goal extraction
    # ========================================================

    def _extract_positive_goal_atoms(
        self,
        expr,
    ):
        """
        Extract positive grounded fluent atoms.

        Supported:

            on(A, B)

        and:

            And(
                on(A, B),
                on(B, C)
            )

        Negative, disjunctive, numeric, etc. goals are
        deliberately rejected in the current Blocks World
        version.
        """

        if expr.is_fluent_exp():
            return [expr]

        if expr.is_and():

            atoms = []

            for arg in expr.args:
                atoms.extend(
                    self
                    ._extract_positive_goal_atoms(
                        arg
                    )
                )

            return atoms

        raise ValueError(
            "Current context encoding "
            "supports only positive "
            "Boolean fluent goals. "
            f"Unsupported goal: {expr}"
        )

    # ========================================================
    # Initial state + goal context encoding
    # ========================================================

    def encode_problem_context(
        self,
        problem,
    ) -> EncodedContext:
        """
        Encode the problem's CURRENT initial state and goal.

        Representation:

            [INIT, NONE, NONE]

            initial fact
            initial fact
            ...
            PAD
            PAD

            [GOAL, NONE, NONE]

            goal fact
            goal fact
            ...
            PAD
            PAD


        Each section receives one slot for every grounded
        fluent in the problem.

        If n is the number of grounded fluents:

            context_rows = 2 + 2n


        Only TRUE initial-state fluents are serialized.

        For the positive STRIPS goal representation used here,
        omitted goal fluents mean "not required", NOT false.
        """

        # ----------------------------------------------------
        # Read initial values once.
        # ----------------------------------------------------

        initial_values = (
            problem.initial_values
        )

        num_ground_fluents = len(
            initial_values
        )

        # ----------------------------------------------------
        # TRUE initial facts
        # ----------------------------------------------------

        initial_facts = []

        for (
            fluent_exp,
            value,
        ) in initial_values.items():

            # Current project assumes Boolean STRIPS fluents.
            if value.is_true():

                initial_facts.append(
                    self.fluent_to_tuple(
                        fluent_exp
                    )
                )

        # ----------------------------------------------------
        # Canonical ordering.
        #
        # Avoid multiple sequence encodings for same logical
        # state.
        # ----------------------------------------------------

        initial_facts.sort(
            key=lambda row:
            tuple(row)
        )

        # ----------------------------------------------------
        # Goal atoms
        # ----------------------------------------------------

        goal_atoms = []

        for goal in problem.goals:

            goal_atoms.extend(
                self
                ._extract_positive_goal_atoms(
                    goal
                )
            )

        goal_facts = [
            self.fluent_to_tuple(
                goal_atom
            )
            for goal_atom
            in goal_atoms
        ]

        goal_facts.sort(
            key=lambda row:
            tuple(row)
        )

        # ----------------------------------------------------
        # Capacity checks
        # ----------------------------------------------------

        if (
            len(initial_facts)
            > num_ground_fluents
        ):
            raise RuntimeError(
                "Initial-state fact count "
                "exceeds number of grounded "
                "fluents."
            )

        if (
            len(goal_facts)
            > num_ground_fluents
        ):
            raise ValueError(
                "Goal fact count exceeds "
                "available grounded-fluent "
                "capacity."
            )

        rows = []
        mask = []

        # ====================================================
        # INITIAL STATE SECTION
        # ====================================================

        rows.append(
            self.init_row()
        )

        mask.append(True)

        for fact in initial_facts:

            rows.append(
                fact
            )

            mask.append(True)

        # Pad state section to exactly n fluent positions.

        for _ in range(
            num_ground_fluents
            - len(initial_facts)
        ):

            rows.append(
                self.pad_row()
            )

            mask.append(False)

        # ====================================================
        # GOAL SECTION
        # ====================================================

        rows.append(
            self.goal_row()
        )

        mask.append(True)

        for fact in goal_facts:

            rows.append(
                fact
            )

            mask.append(True)

        # Pad goal section.

        for _ in range(
            num_ground_fluents
            - len(goal_facts)
        ):

            rows.append(
                self.pad_row()
            )

            mask.append(False)

        # ----------------------------------------------------
        # Convert symbolic rows into categorical IDs.
        # ----------------------------------------------------

        tensor = torch.tensor(
            [
                [
                    self.encode_token(
                        token
                    )
                    for token in row
                ]
                for row in rows
            ],
            dtype=torch.long,
        )

        attention_mask = (
            torch.tensor(
                mask,
                dtype=torch.bool,
            )
        )

        return EncodedContext(
            tensor=tensor,
            attention_mask=attention_mask,
            num_initial_facts=len(
                initial_facts
            ),
            num_goal_facts=len(
                goal_facts
            ),
        )

    # ========================================================
    # Combine context and plan
    # ========================================================

    def combine_context_and_plan(
        self,
        context: EncodedContext,
        plan_tensor: torch.Tensor,
    ) -> tuple[
        torch.LongTensor,
        torch.BoolTensor,
    ]:
        """
        Combine context and plan into one structured tensor.

        Output:

            [
                context
                ----
                plan
            ]

        The context padding mask is preserved.

        Every plan row is considered active because even PAD/END
        are categorical variables the DFM may need to predict.
        """

        plan_tensor = (
            plan_tensor
            .detach()
            .cpu()
        )

        expected_plan_shape = (
            self.num_plan_slots,
            self.tuple_width,
        )

        if (
            tuple(plan_tensor.shape)
            != expected_plan_shape
        ):
            raise ValueError(
                f"Expected plan tensor shape "
                f"{expected_plan_shape}, got "
                f"{tuple(plan_tensor.shape)}."
            )

        combined = torch.cat(
            [
                context.tensor,
                plan_tensor,
            ],
            dim=0,
        )

        plan_mask = torch.ones(
            self.num_plan_slots,
            dtype=torch.bool,
        )

        combined_mask = torch.cat(
            [
                context.attention_mask,
                plan_mask,
            ],
            dim=0,
        )

        return (
            combined,
            combined_mask,
        )

    # ========================================================
    # Flatten / unflatten PLAN tensor
    # ========================================================

    def flatten_plan(
        self,
        tensor: torch.Tensor,
    ) -> torch.LongTensor:
        """
        [H + 1, tuple_width]
            ->
        [(H + 1) * tuple_width]
        """

        return tensor.reshape(-1)

    def unflatten_plan(
        self,
        tensor: torch.Tensor,
    ) -> torch.LongTensor:
        """
        [(H + 1) * tuple_width]
            ->
        [H + 1, tuple_width]
        """

        expected_size = (
            self.num_plan_slots
            * self.tuple_width
        )

        if (
            tensor.numel()
            != expected_size
        ):
            raise ValueError(
                f"Expected {expected_size} "
                f"categorical entries, got "
                f"{tensor.numel()}."
            )

        return tensor.reshape(
            self.num_plan_slots,
            self.tuple_width,
        )

    # ========================================================
    # General flattening helper
    # ========================================================

    @staticmethod
    def flatten(
        tensor: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Flatten any structured categorical tensor.
        """

        return tensor.reshape(-1)

    # ========================================================
    # Vocabulary debugging
    # ========================================================

    def print_vocabulary(
        self,
    ):
        """
        Print categorical vocabulary in ID order.
        """

        for index in range(
            self.vocab_size
        ):

            print(
                f"{index:3d}: "
                f"{self.id_to_token[index]}"
            )
def encode_enumeration_result(
    result,
    codec: PlanCodec,
):
    """
    Convert an enumeration result into PyTorch tensors.

    Assumes the enumeration function already added END/PAD.
    """

    encoded_plans = [
        codec.encode_plan(
            p,
            # already_padded=True,
        ).tensor
        for p in result["plans"]
    ]

    encoded_trajectories = [
        codec.encode_plan(
            p,
            # already_padded=True,
        ).tensor
        for p in result["trajectories"]
    ]

    if encoded_plans:
        plans_tensor = torch.stack(
            encoded_plans
        )
    else:
        plans_tensor = torch.empty(
            (
                0,
                codec.num_plan_slots,
                codec.tuple_width,
            ),
            dtype=torch.long,
        )

    if encoded_trajectories:
        trajectories_tensor = torch.stack(
            encoded_trajectories
        )
    else:
        trajectories_tensor = torch.empty(
            (
                0,
                codec.num_plan_slots,
                codec.tuple_width,
            ),
            dtype=torch.long,
        )

    return {
        "plans": plans_tensor,
        "trajectories": trajectories_tensor,
    }