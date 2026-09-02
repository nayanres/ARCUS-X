# ARCUS-X Updated Taxonomy Documentation

## 1. Output Compliance Layer (`OutputCompliance`)
- `PERFECT_FORMAT`: Output contains only the expected trajectory format without extra text or wrapping.
- `VERBOSE_CORRECT`: Output includes reasoning steps, explanation, or markdown wrapping, but the extracted trajectory perfectly matches ground truth.
- `MALFORMED_OUTPUT`: Trajectory output was corrupted or malformed.
- `NO_TRAJECTORY`: No coordinates could be extracted from the output.
- `PARSE_FAILURE`: Coordinate tokens failed parsing.

## 2. Formatting Subcodes
- `F001_VERBOSE_CORRECT`: Correct trajectory embedded within explanatory text.
- `F002_MARKDOWN_WRAPPER`: Correct trajectory wrapped in markdown code blocks.
- `F003_INVALID_DELIMITERS`: Non-standard delimiters used.
- `F004_MISSING_TRAJECTORY`: Empty trajectory response.
- `F005_MALFORMED_TRAJECTORY`: Corrupted coordinate sequence.
- `F006_PARSE_FAILURE`: Invalid coordinate token format.

## 3. Cognitive Failure Taxonomy (`FailureMechanism`)
- `NONE`: No cognitive failure.
- `STATE_TRACKING_FAILURE`: Lost state after initial correct steps (recursive drift / persistent offset).
- `TRANSITION_FAILURE`: Incorrect transition application (sign inversion, axis swap, magnitude error, parity error).
- `SEMANTIC_FAILURE`: Initial state offset or environment rule interpretation mismatch.
- `OUTPUT_FORMAT_FAILURE`: Formatting prevented extraction or no correct trajectory recovered.
- `HORIZON_FAILURE`: Gradual decay or cliff collapse over increasing horizons.
- `UNKNOWN_FAILURE`: Unclassified failure mode.
