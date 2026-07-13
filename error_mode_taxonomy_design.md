# Error-Mode Taxonomy Pipeline Architecture

## 1. Objective
Automate the classification of model failure modes during cognitive fracture benchmarking by analyzing step-by-step vector deltas (dx_actual, dy_actual) relative to a modulo-wrapped grid dimension. The pipeline must dynamically detect and categorize four distinct error modes without hard-coding permutations:

- **Memory Failure / State Dissipation** – Random coordinate jumps.
- **Algorithm Collapse / Parity Sync Loss** – Valid rules applied out of order.
- **Heuristic Substitution** – Repeating a fixed transform vector for ≥3 steps.
- **Arithmetic Slips** – Minor off-by-one/off-by-variable calculation drift.

## 2. Core Data Structures

| Structure | Purpose | Key Fields |
|-----------|---------|------------|
| `TrajectoryStep` | Represents a single movement in the coordinate grid. | `x: int`, `y: int`, `timestamp: int`, `delta_x: float`, `delta_y: float` |
| `TrajectoryRecord` | Container for ordered sequence of steps up to current depth `z`. | `steps: List[TrajectoryStep]`, `depth: int`, `gravity_target: float` |
| `ErrorModeClassification` | Holds classification result for a given depth. | `mode: str` (one of the four), `confidence: float`, `evidence: List[Tuple[float, float]]` |
| `DepthErrorModeStore` | Depth‑stratified aggregation of error‑mode percentages. | `depth: int`, `mode: str`, `count: int`, `percentage: float` |

All structures are immutable and serialized as JSON for storage.

## 3. Vector Delta Extraction

1. **Step Generation** – After each coordinate update, compute `dx = x_current - x_previous`, `dy = y_current - y_previous`.
2. **Modulo Wrapping** – Apply `dx_mod = dx % grid_dim`, `dy_mod = dy % grid_dim` to normalize movements within the grid bounds.
3. **Delta Buffer** – Maintain a rolling buffer of the last `N` deltas (e.g., `N = 10`) for statistical analysis.

**Function Signature**  
```python
def extract_deltas(trajectory: List[TrajectoryStep], grid_dim: int) -> List[Tuple[float, float]]:
    ...
```

## 4. Error‑Mode Detection Algorithms

### 4.1 Memory Failure / State Dissipation
- **Pattern**: Large, irregular jumps exceeding a threshold relative to typical step magnitude.
- **Algorithm**:
  1. Compute Z‑score of each delta magnitude `sqrt(dx_mod² + dy_mod²)`.
  2. Flag steps where `abs(z_score) > THRESHOLD` (e.g., 3.0) as potential memory failures.
  3. Cluster flagged steps; if ≥1 flagged step within a window, classify the depth as **Memory Failure**.

### 4.2 Algorithm Collapse / Parity Sync Loss
- **Pattern**: Application of a rule out of the expected parity order (even/odd step expectations).
- **Algorithm**:
  1. Encode expected transformation rule for step index `i` (e.g., Rule Alpha for even `i`, Rule Beta for odd `i`).
  2. Compare actual delta direction against expected direction using a directional cosine metric.
  3. If > `M` consecutive steps violate expected parity, classify as **Algorithm Collapse**.

### 4.3 Heuristic Substitution
- **Pattern**: Repeating the same transform vector for ≥3 consecutive steps.
- **Algorithm**:
  1. Slide a window of size `W = 3` over the delta sequence.
  2. For each window, compute the variance of deltas.
  3. If variance < `VARIANCE_THRESHOLD` and the same delta repeats ≥3 times, classify as **Heuristic Substitution**.

### 4.4 Arithmetic Slips
- **Pattern**: Systematic off‑by‑one or off‑by‑variable errors in coordinate calculations.
- **Algorithm**:
  1. Derive the *expected* delta based on the known rule set (e.g., `(+2, +1)` for even steps, `(+1, +3)` for odd steps).
  2. Compare actual delta to expected delta; compute absolute error per component.
  3. If error consistently equals `1` (or a small constant) across many steps, classify as **Arithmetic Slip**.

All detectors return a **confidence score** (0‑1) based on how strongly the pattern matches.

## 5. Integration Points in `fracture_finder.py`

1. **Enhanced `parse_matrix_telemetry`** – After constructing `accuracy_curve`, also capture the raw step‑by‑step coordinate sequence for each depth `z`.
2. **New Method** – `analyze_error_modes(self, depth: int, deltas: List[Tuple[float, float]], grid_dim: int) -> ErrorModeClassification`:
   - Orchestrates the four detectors.
   - Returns the most probable error mode and confidence.
3. **Store Results** – Populate `self.error_mode_history: Dict[int, ErrorModeClassification]`.
4. **Depth‑Stratified Storage** – Maintain `self.depth_error_stats: List[DepthErrorModeStore]` to be serialized with the final report.

## 6. Depth‑Stratified Percentage Distribution Storage

- **Structure**: `self.depth_error_distribution: Dict[int, Dict[str, float]]` where each key is a depth `z` and the value is a dict mapping error‑mode names to their percentage of occurrences at that depth.
- **Update Frequency**: After processing each depth, compute percentages from `self.depth_error_stats` and store.
- **Serialization**: Include in the final JSON report under `"depth_error_distribution"`.

## 7. Computational Overhead Analysis

| Operation | Complexity | Practical Impact |
|-----------|------------|------------------|
| Delta extraction per step | O(1) | Negligible; only a few arithmetic ops. |
| Sliding‑window pattern detection (Heuristic Substitution) | O(k) where *k* = window size (fixed ≤10) | Constant time; no scaling with `z`. |
| Z‑score / statistical outlier detection | O(m) where *m* = buffer length (fixed) | Constant overhead. |
| Aggregation into depth‑stratified dict | O(1) per depth | Minimal. |
| Overall per‑depth cost | **O(1)** (bounded by fixed‑size buffers) | **Trivial** relative to model evaluation; can be performed in‑memory without measurable latency. |

## 8. Architectural Blueprint of the Vector Verification Function

```mermaid
flowchart TD
    A[Input: TrajectoryRecord] --> B[Extract Deltas (dx, dy)]
    B --> C[Apply Modulo Grid Wrapping]
    C --> D[Buffer Recent Deltas]
    D --> E{Error Mode Detection}
    E -->|Memory Failure| F[Classify as Memory Failure]
    E -->|Algorithm Collapse| G[Classify as Algorithm Collapse]
    E -->|Heuristic Substitution| H[Classify as Heuristic Substitution]
    E -->|Arithmetic Slip| I[Classify as Arithmetic Slip]
    F --> J[Update DepthErrorModeStore]
    G --> J
    H --> J
    I --> J
    J --> K[Serialize Distribution for Reporting]
    K --> L[Output: Error Mode + Confidence per Depth]
```

- **Inputs**: `TrajectoryRecord` (step list), `grid_dim`.
- **Processing**: Delta extraction → modular normalization → pattern detection → classification.
- **Outputs**: Error‑mode label, confidence, and updated depth‑stratified statistics.

## 9. Next Steps

- [x] Define data structures for step‑by‑step trajectory storage.
- [x] Design error‑mode detection algorithms for all four modes.
- [ ] Plan integration points in `fracture_finder.py` (modify `parse_matrix_telemetry`, add `analyze_error_modes`).
- [ ] Design storage mechanism for depth‑stratified error‑mode percentages.
- [ ] Analyze computational overhead of vector delta extraction (already determined trivial).
- [ ] Create architectural blueprint document (completed).
- [ ] Review blueprint with user for feedback.

