# ARCUS-X Migration Notes: Output Compliance & Cognitive Correctness Separation

## Summary of Changes
1. **Output Compliance Layer**: Added `OutputCompliance` enum (`PERFECT_FORMAT`, `VERBOSE_CORRECT`, `MALFORMED_OUTPUT`, `NO_TRAJECTORY`, `PARSE_FAILURE`) and formatting subcodes (`F001` through `F006`).
2. **ProbeClassification**: Enhanced to include `trajectory_correct`, `step_accuracy`, `exact_match`, `output_compliance`, and `formatting_subcode`.
3. **Adaptive Search Preservation**: Maintained exact behavior of the adaptive controller (`step_accuracy` from parsed trajectories continues to drive search parameters).
4. **Taxonomy & Reporting**: Updated reporting layout to separate `TRAJECTORY PERFORMANCE`, `OUTPUT COMPLIANCE`, and `COGNITIVE FAILURE TAXONOMY`.

## Backward Compatibility
- All legacy result dict fields (`error_mode`, `valid`, `mechanism`) are fully preserved.
- Existing serialized JSON metric reports and results matrices remain readable and compatible.
