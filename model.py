
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DFMConfig:
    vocab_size: int

    tuple_width: int

    # Number of rows in encoded initial-state + goal context.
    context_rows: int

    # H + 1, because END occupies one extra plan row.
    plan_slots: int

    d_model: int = 1024
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
def sinusoidal_time_embedding(
    t: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Convert t in [0,1] to a sinusoidal embedding.

    Parameters
    ----------
    t:
        Shape [batch]

    Returns
    -------
    Shape [batch, dim]
    """

    if t.ndim != 1:
        raise ValueError(
            f"Expected t shape [B], got {t.shape}"
        )

    half = dim // 2

    device = t.device

    if half == 0:
        return t[:, None]

    frequencies = torch.exp(
        torch.linspace(
            0.0,
            math.log(10000.0),
            half,
            device=device,
        )
    )

    angles = (
        2.0
        * math.pi
        * t[:, None]
        * frequencies[None, :]
    )

    embedding = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=-1,
    )

    if embedding.shape[-1] < dim:
        embedding = F.pad(
            embedding,
            (0, dim - embedding.shape[-1]),
        )

    return embedding

def build_plan_output_mask(
    codec,
) -> torch.BoolTensor:
    """
    Static structural mask for network outputs.

    Shape:

        [plan_slots, tuple_width, vocab_size]

    Operator position may generate:

        action names
        END
        PAD

    Argument positions may generate:

        object names
        NONE
        PAD

    MASK is deliberately NOT a valid clean target.

    This does not enforce action-schema-dependent constraints such
    as:

        pick_up(A, NONE)

    versus

        pick_up(A, B)

    The validator will catch those. They can be added later as
    dynamic masks.
    """

    mask = torch.zeros(
        (
            codec.num_plan_slots,
            codec.tuple_width,
            codec.vocab_size,
        ),
        dtype=torch.bool,
    )

    # --------------------------------------------------------
    # Operator position
    # --------------------------------------------------------

    operator_ids = [
        codec.token_to_id[name]
        for name in codec.action_names
    ]

    operator_ids += [
        codec.end_id,
        codec.pad_id,
    ]

    mask[:, 0, operator_ids] = True

    # --------------------------------------------------------
    # Argument positions
    # --------------------------------------------------------

    argument_ids = [
        codec.token_to_id[name]
        for name in codec.object_names
    ]

    argument_ids += [
        codec.none_id,
        codec.pad_id,
    ]

    for column in range(
        1,
        codec.tuple_width,
    ):
        mask[:, column, argument_ids] = True

    return mask


# ============================================================
# Transformer denoiser
# ============================================================

class PlanningDFM(nn.Module):
    """
    One Transformer models

        p_theta(
            X_1^i
            |
            X_t,
            initial state,
            goal,
            t
        )

    Context is fixed.

    Only the plan participates in the discrete flow.
    """

    def __init__(
        self,
        config: DFMConfig,
        plan_output_mask: torch.BoolTensor | None = None,
    ):
        super().__init__()

        self.config = config

        self.context_tokens = (
            config.context_rows
            * config.tuple_width
        )

        self.plan_tokens = (
            config.plan_slots
            * config.tuple_width
        )

        self.total_tokens = (
            self.context_tokens
            + self.plan_tokens
        )

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # ----------------------------------------------------
        # Absolute position
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            self.total_tokens,
            config.d_model,
        )

        # ----------------------------------------------------
        # Segment:
        #
        # 0 = context
        # 1 = plan
        # ----------------------------------------------------

        self.segment_embedding = nn.Embedding(
            2,
            config.d_model,
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.output_head = nn.Linear(
            config.d_model,
            config.vocab_size,
        )

        # ----------------------------------------------------
        # Static legal-token mask
        # ----------------------------------------------------

        if plan_output_mask is not None:

            expected = (
                config.plan_slots,
                config.tuple_width,
                config.vocab_size,
            )

            if tuple(
                plan_output_mask.shape
            ) != expected:
                raise ValueError(
                    "Wrong plan output mask shape. "
                    f"Expected {expected}, got "
                    f"{tuple(plan_output_mask.shape)}."
                )

            self.register_buffer(
                "plan_output_mask",
                plan_output_mask.bool(),
            )

        else:
            self.plan_output_mask = None

    def forward(
        self,
        context: torch.LongTensor,
        context_mask: torch.BoolTensor,
        x_t: torch.LongTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context:
            [B, context_rows, tuple_width]

        context_mask:
            [B, context_rows]

            True = real context row
            False = padded context row

        x_t:
            [B, plan_slots, tuple_width]

        t:
            [B]

        Returns
        -------
        logits:
            [
                B,
                plan_slots,
                tuple_width,
                vocab_size
            ]
        """

        B = context.shape[0]

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        expected_context = (
            B,
            self.config.context_rows,
            self.config.tuple_width,
        )

        if tuple(context.shape) != expected_context:
            raise ValueError(
                f"Expected context shape "
                f"{expected_context}, got "
                f"{tuple(context.shape)}."
            )

        expected_plan = (
            B,
            self.config.plan_slots,
            self.config.tuple_width,
        )

        if tuple(x_t.shape) != expected_plan:
            raise ValueError(
                f"Expected plan shape "
                f"{expected_plan}, got "
                f"{tuple(x_t.shape)}."
            )

        # ----------------------------------------------------
        # Flatten symbolic tuple structure only at the
        # Transformer boundary.
        # ----------------------------------------------------

        context_flat = context.reshape(
            B,
            -1,
        )

        plan_flat = x_t.reshape(
            B,
            -1,
        )

        tokens = torch.cat(
            [
                context_flat,
                plan_flat,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Basic token embeddings
        # ----------------------------------------------------

        h = self.token_embedding(
            tokens
        )

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        position_ids = torch.arange(
            self.total_tokens,
            device=tokens.device,
        )

        h = (
            h
            + self.position_embedding(
                position_ids
            )[None, :, :]
        )

        # ----------------------------------------------------
        # Context/plan segment embeddings
        # ----------------------------------------------------

        segment_ids = torch.cat(
            [
                torch.zeros(
                    self.context_tokens,
                    dtype=torch.long,
                    device=tokens.device,
                ),
                torch.ones(
                    self.plan_tokens,
                    dtype=torch.long,
                    device=tokens.device,
                ),
            ]
        )

        h = (
            h
            + self.segment_embedding(
                segment_ids
            )[None, :, :]
        )

        # ----------------------------------------------------
        # Add continuous time information only to plan tokens.
        # ----------------------------------------------------

        time_embedding = (
            sinusoidal_time_embedding(
                t,
                self.config.d_model,
            )
        )

        h[
            :,
            self.context_tokens:,
            :
        ] += time_embedding[:, None, :]

        # ----------------------------------------------------
        # Transformer padding mask
        #
        # PyTorch convention:
        #
        # True = IGNORE token
        # ----------------------------------------------------

        context_token_active = (
            context_mask
            .repeat_interleave(
                self.config.tuple_width,
                dim=1,
            )
        )

        plan_token_active = torch.ones(
            (
                B,
                self.plan_tokens,
            ),
            dtype=torch.bool,
            device=tokens.device,
        )

        active = torch.cat(
            [
                context_token_active,
                plan_token_active,
            ],
            dim=1,
        )

        padding_mask = ~active

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        h = self.transformer(
            h,
            src_key_padding_mask=padding_mask,
        )

        h = self.final_norm(h)

        # ----------------------------------------------------
        # Keep only plan outputs.
        # ----------------------------------------------------

        plan_h = h[
            :,
            self.context_tokens:,
            :
        ]

        logits = self.output_head(
            plan_h
        )

        logits = logits.reshape(
            B,
            self.config.plan_slots,
            self.config.tuple_width,
            self.config.vocab_size,
        )

        # ----------------------------------------------------
        # Remove impossible token categories.
        # ----------------------------------------------------

        if self.plan_output_mask is not None:

            logits = logits.masked_fill(
                ~self.plan_output_mask[None],
                float("-inf"),
            )

        return logits

