# Thread 3: Spectral / operator mixing

**Math source:** Fourier analysis (convolution theorem), operator learning (Fourier Neural
Operators), signal processing.

## Motivation

Attention mixes tokens with a learned, content-dependent, all-pairs kernel: O(n^2) in
sequence length, but maximally flexible. The convolution theorem says any *fixed* (or
low-rank-parameterized) global mixing kernel can be applied in O(n log n) via FFT instead
of the O(n^2) all-pairs form. The question is not "is FFT mixing cheaper" (it obviously
is) but "how much expressivity do we actually give up, and is there a predictable
crossover length below which the gap doesn't matter for a given task class." That
crossover, if it exists and is predictable, is the actual design-relevant quantity —
otherwise this is just a known efficiency trick with no new content.

## Architectural hypothesis

A hybrid or purely spectral mixing layer (global convolution via FFT, with a learned
frequency-domain filter, optionally content-modulated) can match dense attention's accuracy
on tasks that don't require *content-dependent* routing (i.e., where the right mixing
pattern is a function of position/distance rather than token identity), and the point where
it starts to lose ground to attention is predictable from how content-dependent the task's
optimal routing is.

## Falsifiable prediction

On the associative-recall task (see `docs/methodology.md`), which by construction requires
content-dependent routing, spectral mixing should underperform attention at matched
compute regardless of sequence length (theory predicts it *cannot* solve this task family
without a content-dependent gating mechanism layered on top). On a task with the same
sequence lengths but where the optimal mixing is position/distance-based (e.g., a synthetic
task built from convolution/smoothing ground truth), spectral mixing should match attention
at a fraction of the FLOPs, with the FLOP ratio matching the theoretical O(n log n) vs
O(n^2) prediction as n grows in the sweep. If spectral mixing matches attention on the
recall task too, or fails to match it on the position-based task, the "content-dependence is
the deciding factor" hypothesis is falsified.

## Minimal experiment

- Implement a small FFT-based global mixing block and a small dense-attention block,
  matched parameter count and (approximately) matched FLOPs at each sequence length.
- Two synthetic task families: associative recall (content-dependent) and a
  distance/convolution-structured task (position-dependent), sequence lengths swept 128 to
  8192.
- Measure accuracy and wall-clock/FLOPs at each length; fit the FLOPs-vs-length curves for
  both blocks and check against the O(n log n) vs O(n^2) prediction.

## Compute budget

Small models, synthetic data, modest sequence lengths — fits well under a GPU-day even
with the sweep across lengths.

## Bitter-lesson check

- Medium risk flagged deliberately: a *fixed*-basis spectral filter is exactly the kind of
  prior that can cap expressivity and get subsumed by scale (the bitter-lesson failure
  mode). The experiment is designed to find that boundary explicitly (content-dependent vs
  not) rather than assume spectral mixing is "efficient, therefore good."
- FFT is extremely hardware-friendly and highly parallel — if the boundary found is useful
  (e.g., many real sequence-modeling subtasks are more position- than content-dependent
  than commonly assumed), the efficiency win is real and compute-scaling-positive.

## Known prior work / risk of reinventing

FNOs, FNet, global convolution/long-conv sequence models (Hyena, S4-adjacent), and older
Fourier-mixing transformer variants already exist. The contribution here is not the
mechanism but the falsification experiment: explicitly measuring the content-dependence
crossover rather than reporting aggregate benchmark numbers, so the result says *when* to
use which mechanism rather than "ours is faster."

## Status

Not yet run.
