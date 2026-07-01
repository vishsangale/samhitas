"""Pre-registered run for thread 2 (docs/threads/02-criticality-guided-init.md): does the
mean-field-derived depth scale xi(sigma_w2) (experiments/harness/meanfield.py) predict the
empirical (sigma_w2, depth) trainability boundary of a plain, unnormalized, tanh MLP
(experiments/models/deep_mlp.py) on modular arithmetic?

Protocol: fixed sigma_b2=0.1 (matches methodology.md's "cost caveat" -- pick one bias
variance, sweep sigma_w2 only, so the whole thing stays a 1-D sweep against a 1-D theory
curve rather than a 2-D grid). sigma_w2 grid brackets the theory's own derived critical
point sigma_w2*=1.9861 on both the ordered (chi_1<1) and chaotic (chi_1>1) sides, chosen so
that matched pairs of points on either side have roughly matched theoretical xi (see
scripts output / commit message for the grid derivation). Depth grid spans the resulting
xi range. Trainability = reaches a fixed target training loss within a fixed step budget,
matched LR grid + seeds for every (sigma_w2, depth) cell (no arm gets extra tuning trials
-- there's only one arm here, the model itself, being probed across its own theory-implied
knob).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import torch
import torch.nn.functional as F

from experiments.harness import meanfield as mf
from experiments.models.deep_mlp import DeepMLP
from experiments.tasks import modular_arith

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread02_criticality_sanity"

P = 17  # modular arithmetic modulus -- small/cheap, per experiments/README.md's intended
         # fast depth-probe task for this thread
HIDDEN = 32
SIGMA_B2 = 0.1
N_TRAIN = 200
N_TEST = 89  # p*p=289 total pairs; 200 train leaves 89 disjoint test pairs
BATCH = 64
TRAIN_STEPS = 150
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS = list(range(3))
TARGET_LOSS = 1.0  # well below uniform-baseline loss ln(17)=2.83; "reached" = learned something
GRAD_EXPLOSION_ABS = 1e2
GRAD_VANISH_ABS = 1e-8

SIGMA_W2_STAR = mf.find_critical_sigma_w2(SIGMA_B2, mf.tanh_phi, mf.tanh_phi_prime)
SIGMA_W2_GRID = [1.2, 1.5, 1.7, 1.85, 1.93, 1.97, SIGMA_W2_STAR, 2.0, 2.05, 2.15, 2.3, 2.6, 3.0]
# Geometric (ratio ~sqrt(2)) rather than the coarser factor-of-2 grid tried during smoke
# testing: an initial 2-point check (depth 8 vs 32, both xi~14-16) found the trainability
# transition falls entirely *between* those two points (8: 100% reach rate, 32: 0%), so a
# coarse power-of-2 grid can't resolve where the boundary actually sits.
DEPTH_GRID = [4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256]


def xi_at(sigma_w2: float) -> float:
    chi, _ = mf.chi_1(sigma_w2, SIGMA_B2, mf.tanh_phi, mf.tanh_phi_prime)
    return mf.depth_scale(chi)


def train_and_eval(sigma_w2, depth, lr, seed, x_train, y_train, x_test, y_test):
    torch.manual_seed(seed)
    model = DeepMLP(modular_arith.input_dim(P), HIDDEN, depth, modular_arith.num_classes(P),
                     sigma_w2, SIGMA_B2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed * 1_000_003)
    final_loss = None
    reached_target = False
    steps_to_target = None
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, x_train.shape[0], (BATCH,), generator=g)
        logits = model(x_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if not reached_target and final_loss <= TARGET_LOSS:
            reached_target = True
            steps_to_target = step + 1
    with torch.no_grad():
        test_logits = model(x_test)
        test_loss = F.cross_entropy(test_logits, y_test).item()
        test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()

    grad_norm = None
    for p in model.layers[0].parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            break
    numerically_broken = (
        final_loss != final_loss  # NaN
        or (grad_norm is not None and (grad_norm > GRAD_EXPLOSION_ABS or grad_norm < GRAD_VANISH_ABS))
    )

    return {
        "sigma_w2": sigma_w2, "depth": depth, "lr": lr, "seed": seed,
        "final_train_loss": final_loss, "reached_target": reached_target,
        "steps_to_target": steps_to_target, "test_loss": test_loss, "test_acc": test_acc,
        "first_layer_grad_norm": grad_norm, "numerically_broken": numerically_broken,
    }


def main(sigma_w2_grid=None, depth_grid=None, tag="full"):
    sigma_w2_grid = sigma_w2_grid or SIGMA_W2_GRID
    depth_grid = depth_grid or DEPTH_GRID
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = modular_arith.make_dataset(P, seed=0, n_train=N_TRAIN, n_test=N_TEST)

    results = []
    total = len(sigma_w2_grid) * len(depth_grid) * len(LR_GRID) * len(SEEDS)
    done = 0
    t0 = time.time()
    for sigma_w2 in sigma_w2_grid:
        for depth in depth_grid:
            for lr in LR_GRID:
                for seed in SEEDS:
                    r = train_and_eval(sigma_w2, depth, lr, seed, x_train, y_train, x_test, y_test)
                    results.append(r)
                    done += 1
            elapsed = time.time() - t0
            # per-(sigma_w2,depth) cell summary line, after its lr x seed block finishes
            cell = [r for r in results if r["sigma_w2"] == sigma_w2 and r["depth"] == depth]
            best_rate = max(
                sum(1 for r in cell if r["lr"] == lr_ and r["reached_target"]) / len(SEEDS)
                for lr_ in LR_GRID
            )
            print(f"[{done}/{total}] sigma_w2={sigma_w2:.4f} depth={depth:4d} "
                  f"xi={xi_at(sigma_w2):8.2f} best_reach_rate={best_rate:.2f} elapsed={elapsed:.0f}s")

    out_path = RUN_DIR / f"results_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
