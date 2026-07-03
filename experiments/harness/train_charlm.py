"""Char-LM training loop for thread 6's real run (baseline MLP arm + thread-1 recurrence
arm), producing the SAME `RunResult` schema as harness/train.py so that harness/report.py's
`save_run` and `summarize_sweep` work on its output unchanged -- ingestion of the GPU run is
then "drop the JSON into experiments/runs/<name>/ and run summarize_sweep", no new format.

Kept separate from train.py's `train_one` rather than folding into it: train.py is hardwired
to MuPMLP + modular_arith and is already validated; this adds the char-LM task/models without
touching that path. Only the `RunResult` dataclass is imported and reused.

Metric, matching train.py: steps-to-reach-target-train-loss across several thresholds (not a
single final-loss argmin), for the reasons train.py's docstring gives. Thresholds here are
in nats and sized for char-LM (init cross-entropy is log(vocab) ~= 4.17); see
thread06_charlm_smoke.py for the pilot that set them.
"""

import time
from dataclasses import dataclass, asdict, field

import torch
import torch.nn.functional as F

from experiments.harness.train import RunResult
from experiments.models.char_lm_mlp import CharLMMLP
from experiments.models.char_lm_recurrence import CharLMRecurrence
from experiments.tasks import char_lm


@dataclass
class CharLMConfig:
    arm: str               # "mlp" or "recurrence"
    parametrization: str   # "sp" or "mup"
    width: int
    base_width: int
    base_lr: float
    seed: int
    context_len: int = 32
    d_embed: int = 64      # mlp arm only (fixed featurizer dim, does NOT scale with width)
    depth: int = 4         # mlp arm only (proj + depth-2 hidden + output)
    eps: float = 0.05      # recurrence arm only (spectral radius = 1-eps)
    batch_size: int = 64
    eval_batch_size: int = 512
    max_steps: int = 2000
    target_train_losses: list = field(default_factory=lambda: [3.0, 2.7, 2.4])


def build_model(cfg: CharLMConfig):
    vocab = char_lm.vocab_size()
    if cfg.arm == "mlp":
        return CharLMMLP(vocab, cfg.width, cfg.context_len, cfg.d_embed, cfg.depth,
                         cfg.base_width, cfg.parametrization)
    if cfg.arm == "recurrence":
        return CharLMRecurrence(vocab, cfg.width, cfg.eps, cfg.base_width, cfg.parametrization)
    raise ValueError(f"unknown arm {cfg.arm!r}")


def charlm_flops(cfg: CharLMConfig, model, steps_run: int) -> int:
    """Analytic FLOP count, method stated per docs/methodology.md.

    Same 6*fan_in*fan_out-per-sample convention as harness/flops.py (multiply-add = 2,
    backward ~= 2x forward). Two arms:

    - mlp: proj (6*context*d_embed*W) + (depth-2) hidden (6*W*W) + output (6*W*vocab),
      all per-sample; embedding lookup ignored (negligible, like flops.py ignores bias).
    - recurrence: B and the A-scan are each applied at every one of `context_len` timesteps,
      so 2 * seq_len * 6*W*W per sample, plus readout 6*W*vocab per sample. On TOP of the
      per-sample matmuls there is one matrix_exp(WxW) per forward step (NOT per sample):
      torch.matrix_exp uses scaling-and-squaring, ~O(width^3). We charge it a hand-picked
      constant C_EXPM=60 * W^3 per step (fwd+bwd, order-of-magnitude only -- the exact
      squaring count is norm-dependent). This term dominates at large width (W^3 vs W^2),
      and is FLAGGED as approximate: wall-clock, not this count, is the load-bearing hardware
      number per methodology.
    """
    W, vocab, C = cfg.width, model.vocab, cfg.context_len
    if cfg.arm == "mlp":
        per_sample = 6 * C * cfg.d_embed * W
        per_sample += (cfg.depth - 2) * 6 * W * W
        per_sample += 6 * W * vocab
        return per_sample * cfg.batch_size * steps_run
    # recurrence
    per_sample = 2 * C * 6 * W * W + 6 * W * vocab
    C_EXPM = 60
    per_step_expm = C_EXPM * (W ** 3)
    return (per_sample * cfg.batch_size + per_step_expm) * steps_run


def train_one_charlm(cfg: CharLMConfig, log_every: int = 50, device: str = "cpu") -> RunResult:
    torch.manual_seed(cfg.seed)
    train_data, test_data = char_lm.make_split()
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.param_groups(cfg.base_lr))

    # Training-batch RNG, decorrelated from any dataset seed (same convention as train.py).
    g = torch.Generator().manual_seed(cfg.seed * 1000 + 1)
    # Fixed eval batches (drawn once per run), so final_* metrics aren't confounded by which
    # random windows happened to be sampled at the end of training.
    g_eval_test = torch.Generator().manual_seed(cfg.seed * 7 + 3)
    x_eval, y_eval = char_lm.make_batch(test_data, cfg.context_len, cfg.eval_batch_size, g_eval_test)
    x_eval, y_eval = x_eval.to(device), y_eval.to(device)
    g_eval_train = torch.Generator().manual_seed(cfg.seed * 13 + 5)
    x_tr_eval, y_tr_eval = char_lm.make_batch(train_data, cfg.context_len, cfg.eval_batch_size, g_eval_train)
    x_tr_eval, y_tr_eval = x_tr_eval.to(device), y_tr_eval.to(device)

    loss_curve = []
    steps_to_target = {t: None for t in cfg.target_train_losses}
    pending = set(cfg.target_train_losses)
    start = time.perf_counter()
    step = 0
    for step in range(cfg.max_steps):
        xb, yb = char_lm.make_batch(train_data, cfg.context_len, cfg.batch_size, g)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if step % log_every == 0 or step == cfg.max_steps - 1:
            loss_curve.append((step, loss_val))
        for t in list(pending):
            if loss_val < t:
                steps_to_target[t] = step
                pending.discard(t)
        if not pending:
            break
    wall_clock = time.perf_counter() - start
    steps_run = step + 1

    with torch.no_grad():
        final_train_loss = F.cross_entropy(model(x_tr_eval), y_tr_eval).item()
        eval_logits = model(x_eval)
        final_test_loss = F.cross_entropy(eval_logits, y_eval).item()
        final_test_acc = (eval_logits.argmax(dim=-1) == y_eval).float().mean().item()

    flops = charlm_flops(cfg, model, steps_run)

    return RunResult(
        config=asdict(cfg),
        steps_to_target=steps_to_target,
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
        final_test_acc=final_test_acc,
        wall_clock_seconds=wall_clock,
        flops=flops,
        loss_curve=loss_curve,
    )
