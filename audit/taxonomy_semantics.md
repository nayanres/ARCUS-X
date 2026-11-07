# Taxonomy `None` vs. Exact Match

## Conclusion

The repository does not define `None` as full trajectory correctness. The
reported label is a composite of two policies:

1. The classifier's primary mechanism is `None` when it detects no transition
   rule mismatch and, for the direct success branch, the predicted coordinates
   exactly equal ground truth.
2. The reporting layer canonicalizes a result to `None` when **any** of these
   hold: result-level Exact Match, classifier Exact Match, classifier
   `trajectory_correct`, or result-level `step_accuracy >= 1.0`.

The fourth condition deliberately permits `None` when the requested transitions
score perfectly but the emitted path is not the requested full trajectory. It
does not mean all emitted coordinates are correct, nor does it mean full
horizon compliance. It is closest to "no error on the requested transitions
counted by Step Accuracy," with additional classifier and parser edge cases
described below.

## Implemented conditions

- [`is_fully_correct`](../arcus/analysis/reporting.py#L44) includes
  `step_accuracy >= 1.0`; [`canonicalize_taxonomy`](../arcus/analysis/reporting.py#L64)
  turns that result into `None`.
- [`compare_trajectories`](../arcus/evaluation/metrics.py#L61) excludes the
  supplied initial state from Step Accuracy and compares only the overlap of
  predicted and requested paths. Extra emitted coordinates are not part of the
  Step Accuracy numerator. Exact Match separately requires equal path lengths
  and every coordinate to match.
- In the classifier, no mismatched transition deltas plus a length mismatch
  yields `Horizon Failure`; no mismatched deltas plus equal lengths yields
  mechanism `None`. When there are transition mismatches, other precedence
  rules apply first; an immediate divergence with increasing drift can also
  yield `Horizon Failure`. Initial-state mismatch is handled as Semantic
  Failure before those checks. See
  [`_attribute_mechanism`](../arcus/analysis/taxonomy.py#L528).
- `Horizon Match` and the continuous `HorizonCompliance` score are separate
  from Exact Match and Step Accuracy
  ([`metrics.py`](../arcus/evaluation/metrics.py#L140),
  [`HorizonCompliance`](../arcus/evaluation/metrics.py#L148)).

Therefore, among the choices in the audit question, the best answer is **some
other explicitly implemented condition**: the classifier may return `None`
when no transition-delta mismatch is detected and lengths agree, while the
reporting layer may also assign `None` for exact/trajectory correctness or
perfect scored transition accuracy. It is not equivalent to full trajectory
correctness. It is only approximately "no classified error among emitted
transitions," because transition comparison can be modulo-based, Step Accuracy
excludes the initial coordinate, and extra coordinates after the requested
horizon are not scored.

## Evidence that the reporting override is intentional

The reporting helper and an explicit regression test were introduced in
commit `5ce21a7d` ("Add reporting module and update output files for analysis
results"). The test
[`test_fully_correct_probe_overrides_stale_failure_label`](../tests/test_post_run_analyzer.py#L16)
sets Step Accuracy to 1.0 while Exact Match and `trajectory_correct` are false,
then expects a `Horizon Collapse` label to become `None`. A second test
asserts this policy in aggregated taxonomy percentages. This establishes that
the behavior is intentional in the repository implementation; the commit
message and tests do not provide a rationale for its scientific interpretation.

## Probe examples and prevalence

Counts below use deduplicated task-condition probes and valid-response
denominators from [`taxonomy_model_summary.csv`](./taxonomy_model_summary.csv).

| Model | Valid probes | Reported `None` | Exact Match | `None` but not exact | Step-perfect override among those |
|---|---:|---:|---:|---:|---:|
| Mistral Large | 3,280 | 1,501 (45.76%) | 37 (1.13%) | 1,464 | 1,464 |
| Kimi K2.5 | 6,128 | 3,231 (52.73%) | 2,539 (41.43%) | 692 | 691 |
| Ministral 14B | 4,092 | 1,132 (27.66%) | 698 (17.06%) | 434 | 433 |
| Ministral 3B | 1,795 | 274 (15.26%) | 223 (12.42%) | 51 | 50 |

Two clear horizon-override examples:

- Mistral Large, matrix key `219_20_0.5_0_default_0`: at requested horizon
  `z=20`, it emitted 104 coordinates against 21 ground-truth coordinates.
  Step Accuracy is 1.0, Exact Match is false, and Horizon Compliance is
  0.1942. The classifier itself assigns Horizon Failure / Horizon Collapse
  because the matching prefix has the wrong length; reporting changes that
  label to `None`.
- Kimi K2.5, matrix key `219_20_0.5_0_default_0`: it emitted 42 coordinates
  against 21 ground-truth coordinates. Step Accuracy is 1.0, Exact Match is
  false, and Horizon Compliance is 0.4878. It follows the same classifier then
  reporting sequence.

These are intentional separations of transition accuracy from trajectory
completion, not evidence that the extra emitted coordinates were correct.

## Non-override `None` cases

The audit also found one valid, non-exact, sub-perfect-Step-Accuracy `None`
case each for Ministral 14B, Ministral 3B, and Kimi K2.5. They are separated
from the intentional override count in the summary CSV.

- **Ministral 14B**, `2190_70_3.0_0_default_2`: length 71/71, Step Accuracy
  0.5429, first divergence at trajectory position 37 (`[4,10]` vs. `[4,0]`).
  In the raw text, the model later emits a bare `Actions:` line. The post-run
  parser treats that as structured metadata even though it is inside the model
  output section, overwriting the actual action list with an empty value
  ([`post_run_analyzer.py`](../post_run_analyzer.py#L1147)). The classifier
  consequently has no transition actions to check and returns mechanism
  `None`. Reclassifying the same paths with the actual raw action list yields
  State Tracking Failure / Persistent Offset. This is a **post-run parser /
  classification defect**, not the Step Accuracy override. The live runner
  passes actions and transition rules directly to the classifier
  ([`benchmark_runner.py`](../arcus/experiments/benchmark_runner.py#L2000)),
  so this post-run parsing defect should not be assumed to affect live-run
  stored classifications without checking which artifact produced a table.
- **Ministral 3B**, `219_31_3.0_0_default_0`: length 32/32, Step Accuracy
  0.9677, with predicted `[4,10]` versus ground-truth `[4,0]` on a 5-by-10
  grid. The predicted coordinate is outside the canonical grid range but
  equivalent modulo the grid height. The transition classifier accepts its
  wrapped delta and assigns mechanism `None`.
- **Kimi K2.5**, `2190_55_15.0_3_default_4`: length 56/56, Step Accuracy
  0.9818, with predicted `[11,1]` versus ground-truth `[5,1]` on a 6-by-12
  grid. The x-coordinate is outside the canonical range but equivalent modulo
  the grid width; the classifier again accepts the wrapped transition.

The latter two are consistent with a transition-level, toroidal interpretation
if non-canonical coordinates are allowed to denote their modulo-equivalent
states. They are not Exact Matches under the coordinate-string metric, and the
repository does not clearly specify whether such coordinates should be
normalized, accepted, or treated as formatting failures. The classifier's
"Predicted trajectory exactly matches ground truth" explanation is inaccurate
for these cases. Thus they expose a terminology/contract ambiguity, not
necessarily a defect in the transition rule decision itself.

## Documentation and interpretability

The README calls the taxonomy trajectory-level and describes `None` using
"Strict Correctness," but it does not disclose the `step_accuracy >= 1.0`
override. Those descriptions do not adequately communicate the implementation
semantics. The manuscript or publication source is not present in this
checkout, so its definition and any claimed interpretation cannot be verified;
it would be unjustified to call the published wording erroneous without that
source.

Taxonomy shares remain interpretable only as **the distribution of current
reported labels among valid probes**, after the Step Accuracy override and
classifier behavior above. They are not interpretable as full-trajectory
success/failure proportions. For example, Mistral Large's 45.76% reported
`None` versus 1.13% Exact Match is mostly matching-prefix over-generation.
The three sub-perfect `None` cases also show that not every reported `None`
comes from the intentional override. A paper should report Exact Match and
horizon compliance alongside taxonomy proportions and define whether
modulo-equivalent non-canonical coordinates count as correct.

Relevant checked-in public descriptions are in
[`README.md`](../README.md#L135),
[`README.md`](../README.md#L455), and
[`README.md`](../README.md#L815). Probe-level rows, including the classifier
labels and horizon scores, are in
[`probe_level_audit.csv`](./probe_level_audit.csv).
