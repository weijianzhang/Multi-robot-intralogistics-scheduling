# Reproducibility

- All benchmark inputs and seeds come from `benchmark_common.py`. Each case uses
  one fixed instance and different search seeds over repetitions.
- `VNS-HE.py` applies Lemma 2 at decoding. Original `VNS.py` groups priorities
  using `round(priority / 10)` and resolves ties by task ID. These implementations
  also differ in search settings; their comparison is not a decoder-only ablation.
- The four runners in `full_benchmark_lemma2_experiments` replace both full and
  partial candidate evaluation, so initialization, removal gains, reconstruction,
  and nested search use the same sequencing rule. Parameters are inherited,
  not separately retuned for HE.
- VNS-HE restricts assignment to its configured candidate ADR lists. The HE
  baseline decoder enforces capacity eligibility without the same shortlist.
  Common sequencing does not mean every search restriction is identical.
- The adjusted bound uses `min_k(alpha_ik + (q_ik - q_min) * t0)` over eligible
  ADRs. Eligibility is removed in the auxiliary relaxation, not the original
  scheduling problem. `lower_bound.py` delegates to this shared implementation.
- Exported JSON files are audit inputs; batch runners regenerate them using the
  same seed schedule rather than reading the exports automatically.
- Time limits are wall-clock budgets. Expensive initialization and nested search
  may overrun a nominal deadline. Preserve measured elapsed times and use the same
  initialization accounting across methods. Concurrent runs compete for resources.
- Original and HE-enabled outputs have different roots. Check decoder, seeds,
  source/configuration, and successful-run counts before assembling comparisons.
- Plot recorded timestamps and incumbents directly. Do not synthesize common
  starting values or early improvements. The old start-adjusted convergence plot
  and its alternative solver have been excluded from this release.
- Retained statistical tools target the original comparison. A common-HE analysis
  needs its own complete table and recomputed tests. Distinguish repeated searches
  from independent cases; lower means alone do not establish lower variance or
  statistical significance.
- Verify license, author approval, environment versions, and manuscript citation
  before publishing. Syntax checks do not reproduce numerical experiments.
