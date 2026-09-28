import torch
import torch.nn.functional as F

from .model import PlanningDFM


# ============================================================
# Probability schedule
# ============================================================

class LinearSchedule:
    """
    kappa_t = t

    dot{kappa}_t = 1

    This is the simplest DFM scheduler.
    """

    @staticmethod
    def kappa(
        t: torch.Tensor,
    ) -> torch.Tensor:
        return t

    @staticmethod
    def dkappa(
        t: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones_like(t)


# ============================================================
# Forward probability path
# ============================================================

def sample_conditional_path(
    x_1: torch.LongTensor,
    t: torch.Tensor,
    mask_id: int,
    schedule=LinearSchedule,
) -> torch.LongTensor:
    """
    Sample X_t from

        p_t(x^i | x_0, x_1)
        =
        (1-kappa_t) delta_MASK
        +
        kappa_t delta_x1

    independently for each categorical coordinate.

    Parameters
    ----------
    x_1:
        Clean target plan.

        [B, plan_slots, tuple_width]

    t:
        [B]

    Returns
    -------
    x_t:
        Same shape as x_1.
    """

    B = x_1.shape[0]

    kappa = schedule.kappa(t)

    kappa = kappa.reshape(
        B,
        1,
        1,
    )

    reveal_target = (
        torch.rand(
            x_1.shape,
            device=x_1.device,
        )
        < kappa
    )

    x_0 = torch.full_like(
        x_1,
        mask_id,
    )

    x_t = torch.where(
        reveal_target,
        x_1,
        x_0,
    )

    return x_t

@torch.no_grad()
def sample_dfm(
    model: PlanningDFM,
    context: torch.LongTensor,
    context_mask: torch.BoolTensor,
    codec,
    *,
    num_steps: int = 128,
    schedule=LinearSchedule,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
):
    """
    Generate a plan using the DFM probability velocity.

    Starts from:

        X_0 = all MASK

    and repeatedly applies

        delta_Xt
        +
        h * u_t

    where

        u_t =
        dot{kappa}/(1-kappa)
        (p_theta - delta_Xt).

    Equivalently:

        transition distribution
        =
        (1-alpha) delta_Xt
        +
        alpha p_theta

    with

        alpha =
        h * dot{kappa}/(1-kappa).
    """

    if num_steps <= 0:
        raise ValueError(
            "num_steps must be positive."
        )

    if device is None:

        device = next(
            model.parameters()
        ).device

    device = torch.device(device)

    model.eval()

    # --------------------------------------------------------
    # Add batch dimension if necessary.
    # --------------------------------------------------------

    single_sample = (
        context.ndim == 2
    )

    if single_sample:

        context = context.unsqueeze(0)

        context_mask = (
            context_mask
            .unsqueeze(0)
        )

    context = context.to(device)

    context_mask = (
        context_mask.to(device)
    )

    B = context.shape[0]

    # --------------------------------------------------------
    # X_0 = fully masked plan
    # --------------------------------------------------------

    x = torch.full(
        (
            B,
            codec.num_plan_slots,
            codec.tuple_width,
        ),
        codec.mask_id,
        dtype=torch.long,
        device=device,
    )

    h = 1.0 / num_steps

    # --------------------------------------------------------
    # Euler integration
    # --------------------------------------------------------

    for step in range(
        num_steps
    ):

        t_value = (
            step
            / num_steps
        )

        t = torch.full(
            (B,),
            t_value,
            device=device,
        )

        logits = model(
            context=context,
            context_mask=context_mask,
            x_t=x,
            t=t,
        )

        if temperature != 1.0:

            logits = (
                logits
                / temperature
            )

        posterior = F.softmax(
            logits,
            dim=-1,
        )

        kappa = schedule.kappa(t)

        dkappa = schedule.dkappa(t)

        rate = (
            dkappa
            /
            (1.0 - kappa)
        )

        alpha = h * rate

        # ----------------------------------------------------
        # For Euler sampling to define a valid probability
        # distribution, alpha must not exceed one.
        #
        # Linear schedule + uniform steps automatically obeys
        # this.
        # ----------------------------------------------------

        if (
            alpha.max().item()
            > 1.0 + 1e-6
        ):
            raise ValueError(
                "Sampling step is too large for "
                "this scheduler: "
                f"max alpha={alpha.max().item():.4f}. "
                "Increase num_steps."
            )

        alpha = alpha.reshape(
            B,
            1,
            1,
            1,
        )

        # ----------------------------------------------------
        # q =
        #   (1-alpha) delta_current
        #   +
        #   alpha * p_theta
        # ----------------------------------------------------

        transition = (
            alpha
            * posterior
        )

        stay_probability = (
            1.0 - alpha
        )

        transition.scatter_add_(
            dim=-1,
            index=x.unsqueeze(-1),
            src=stay_probability.expand(
                B,
                codec.num_plan_slots,
                codec.tuple_width,
                1,
            ),
        )

        # Numerical safety.
        transition = torch.clamp(
            transition,
            min=0.0,
        )

        transition = (
            transition
            /
            transition.sum(
                dim=-1,
                keepdim=True,
            )
        )

        # ----------------------------------------------------
        # Independently sample each categorical coordinate.
        # ----------------------------------------------------

        flat_distribution = (
            transition.reshape(
                -1,
                codec.vocab_size,
            )
        )

        sampled = torch.multinomial(
            flat_distribution,
            num_samples=1,
        )

        x = sampled.reshape(
            B,
            codec.num_plan_slots,
            codec.tuple_width,
        )

    if single_sample:
        x = x.squeeze(0)

    return x