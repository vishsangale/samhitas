"""Shared training loop: trains one (parametrization, width, lr, seed) config on the
modular-arithmetic task and returns metrics, wall-clock, and FLOPs.

Per docs/methodology.md: every run reports config + raw metrics, both FLOPs and measured
wall-clock, and is seeded explicitly so multi-seed aggregation is possible upstream.
"""

import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from experiments.harness.flops import train_flops
from experiments.models.mlp import MuPMLP
from experiments.tasks import modular_arith


@dataclass
class RunConfig:
    parametrization: str  # "sp" or "mup"
    width: int
    base_width: int
    base_lr: float
    seed: int
    p: int = 17
    depth: int = 3
    n_train: int = 4096
    n_test: int = 1024
    batch_size: int = 128
    n_steps: int = 300


@dataclass
class RunResult:
    config: dict
    final_train_loss: float
    final_test_loss: float
    final_test_acc: float
    wall_clock_seconds: float
    flops: int
    loss_curve: list  # (step, train_loss) sampled every log_every steps


def train_one(cfg: RunConfig, log_every: int = 30) -> RunResult:
    torch.manual_seed(cfg.seed)

    x_train, y_train, x_test, y_test = modular_arith.make_dataset(
        cfg.p, cfg.seed, cfg.n_train, cfg.n_test
    )
    model = MuPMLP(
        input_dim=modular_arith.input_dim(cfg.p),
        width=cfg.width,
        num_classes=modular_arith.num_classes(cfg.p),
        depth=cfg.depth,
        base_width=cfg.base_width,
        parametrization=cfg.parametrization,
    )
    optimizer = torch.optim.Adam(model.param_groups(cfg.base_lr))

    g = torch.Generator().manual_seed(cfg.seed + 1)
    loss_curve = []
    start = time.perf_counter()
    for step in range(cfg.n_steps):
        idx = torch.randint(0, x_train.shape[0], (cfg.batch_size,), generator=g)
        xb, yb = x_train[idx], y_train[idx]

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == cfg.n_steps - 1:
            loss_curve.append((step, loss.item()))
    wall_clock = time.perf_counter() - start

    with torch.no_grad():
        train_logits = model(x_train)
        final_train_loss = F.cross_entropy(train_logits, y_train).item()
        test_logits = model(x_test)
        final_test_loss = F.cross_entropy(test_logits, y_test).item()
        final_test_acc = (test_logits.argmax(dim=-1) == y_test).float().mean().item()

    flops = train_flops(model, cfg.batch_size, cfg.n_steps)

    return RunResult(
        config=asdict(cfg),
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
        final_test_acc=final_test_acc,
        wall_clock_seconds=wall_clock,
        flops=flops,
        loss_curve=loss_curve,
    )
