import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import PlanningDFM
from flow import LinearSchedule, sample_conditional_path

# ============================================================
# DFM training loss
# ============================================================

def dfm_loss(
    logits: torch.Tensor,
    x_1: torch.LongTensor,
) -> torch.Tensor:
    """
    Standard categorical denoiser loss.

    The paper's simple probability-denoiser objective is
    cross entropy against X_1 at every coordinate.
    """

    V = logits.shape[-1]

    return F.cross_entropy(
        logits.reshape(-1, V),
        x_1.reshape(-1),
    )

def train_dfm(
    model: PlanningDFM,
    train_loader: DataLoader,
    codec,
    *,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device=None,
    schedule=LinearSchedule,
    grad_clip: float = 1.0,
):
    """
    Train the DFM probability denoiser.
    """

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    print(f"Training on {device}.")
    device = torch.device(device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        total_examples = 0

        masked_correct = 0
        masked_total = 0

        for batch in train_loader:

            context = (
                batch["context"]
                .to(device)
            )

            context_mask = (
                batch["context_mask"]
                .to(device)
            )

            x_1 = (
                batch["plan"]
                .to(device)
            )

            B = x_1.shape[0]

            # ------------------------------------------------
            # Random flow time
            # ------------------------------------------------

            t = torch.rand(
                B,
                device=device,
            )

            # ------------------------------------------------
            # Sample intermediate plan X_t
            # ------------------------------------------------

            x_t = sample_conditional_path(
                x_1=x_1,
                t=t,
                mask_id=codec.mask_id,
                schedule=schedule,
            )

            # ------------------------------------------------
            # Predict clean target posterior
            # ------------------------------------------------

            logits = model(
                context=context,
                context_mask=context_mask,
                x_t=x_t,
                t=t,
            )

            loss = dfm_loss(
                logits,
                x_1,
            )

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            total_loss += (
                loss.item()
                * B
            )

            total_examples += B

            with torch.no_grad():

                prediction = (
                    logits.argmax(
                        dim=-1
                    )
                )

                # Accuracy only where X_t was still MASK.
                # This is more informative than counting tokens
                # already revealed to the model.
                mask_positions = (
                    x_t
                    != x_1
                )

                if mask_positions.any():

                    masked_correct += (
                        prediction[
                            mask_positions
                        ]
                        ==
                        x_1[
                            mask_positions
                        ]
                    ).sum().item()

                    masked_total += (
                        mask_positions
                        .sum()
                        .item()
                    )

        average_loss = (
            total_loss
            / max(
                total_examples,
                1,
            )
        )

        masked_accuracy = (
            masked_correct
            / max(
                masked_total,
                1,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "loss": average_loss,
                "masked_accuracy":
                    masked_accuracy,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={average_loss:.4f} | "
            f"masked accuracy="
            f"{masked_accuracy:.4f}"
        )

    return history