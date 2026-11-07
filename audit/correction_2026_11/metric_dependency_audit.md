# Metric dependency audit

| Metric or output | Depends on trajectory/reference | Depends on action/rule metadata | Correction status |
|---|---|---|---|
| Step Accuracy | Yes; exact requested-state matches on requested transitions | No | Recomputed independently from preserved predicted and reference paths; unchanged by taxonomy-only and metadata-parser repairs. |
| Exact Match | Yes; literal full path and equal length | No | Recomputed independently; separate from Step Accuracy and taxonomy `None`. |
| Continuity / first divergence | Yes; ordered path comparison | No | Recomputed from raw paths; unaffected by action labels. |
| Horizon Match / Compliance | Yes; generated versus requested path length | No | Recomputed; extra coordinates remain non-compliant. |
| Operational validity | Raw parse / provider markers | No | Recomputed; full-length modulo-equivalent noncanonical paths are parse-valid formatting failures unless unwrapped transition violations retain Transition Failure precedence. |
| Taxonomy | Predicted/reference paths | Yes; transition rules/actions | Corrected for strict `None`, trusted metadata parsing, and noncanonical modulo-equivalent states. |
| AFS live acquisition | Saved probe accuracy/validity and runner search decisions | No taxonomy metadata | Existing acquisition decisions are not retroactively changed; no new model calls were made. |
| Post-hoc fracture depth | Horizon-level accuracy curve and validity policy | No | Sensitivity curves regenerated for valid-only and invalid-as-zero policies; distinct from AFS. |
| Temporal segmentation | Ordered path/reference comparisons and original thirds rule | No | Recomputed independently for available horizons; no taxonomy dependence. |
| CRI / other composite indices | Component metrics and their aggregation | Possibly indirectly | No new manuscript-level claim made; saved snapshots retained as source, and changed taxonomy does not redefine component metrics. |
| Generation overhead / GBI | Token count and expected-token convention | No | Raw token metrics are retained; not changed by taxonomy correction. |

Coordinate contract: the generator specifies `x in [0,width-1]`, `y in [0,height-1]`, a toroidal transition, and a full list containing the initial state plus one coordinate per action. Primary Analysis A uses literal canonical coordinates. A modulo-equivalent noncanonical path is retained as a failure, not `None`; modulo equality is reported only as an Analysis B sensitivity.
