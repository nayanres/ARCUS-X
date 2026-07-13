# AI COGNITIVE FRACTURE BENCHMARKER

This production-grade benchmarking framework evaluates how large language models maintain short-term state during complex, state-dependent iterative tasks. It measures model performance across increasing sequence depths while testing under different contextual conditions.

---

# THE STATE MACHINE CHALLENGE

The benchmark tasks models with tracing an alternating discrete path starting at `[0,0]` up to a specified depth threshold (z-value).

* **Even Steps (0, 2, 4, 6...)**

  * Apply the transformation:

    * Δx = +2
    * Δy = +1

* **Odd Steps (1, 3, 5, 7...)**

  * Apply the transformation:

    * Δx = +1
    * Δy = +3

* **Grid Wrapping**

  * Coordinates wrap using modulo arithmetic based on the grid dimensions:

```
x = (x + Δx) mod grid_dim
y = (y + Δy) mod grid_dim
```

The benchmark evaluates whether the generated path remains consistent with the alternating transition rules as the requested sequence depth increases.

---

# CORE SYSTEM ARCHITECTURE

The codebase is organized into four core modules.

## 1. quickstart_api.py

The primary entry point.

Responsibilities include:

* Parsing CLI arguments
* Orchestrating benchmark execution
* Converting flat telemetry payloads into sequential arrays
* Rendering the final analytics summary

## 2. benchmark_runner.py

The execution engine.

Responsibilities include:

* Adaptive, unbounded bisection search for determining the maximum successfully completed sequence depth
* Path continuity validation
* Detection and logging of invalid path transitions as `STRICT PATH FRAUD`

## 3. model_evaluation.py

The inference layer.

Responsibilities include:

* Managing remote API providers
* Managing local vLLM inference
* Five-attempt exponential backoff retry logic for rate limits and transient network failures

## 4. fracture_finder.py

The analytics layer.

Responsibilities include:

* Parsing telemetry matrices
* Measuring performance across evaluated depths
* Calculating the Cognitive Volatility Score from observed performance deltas

---

# ERROR TOLERANCE AND DESIGN PHILOSOPHY

The verification engine is designed to balance strict rule validation with robustness against minor formatting variation.

It includes:

* An 85% positional-match tolerance
* Sliding-window lookup for token alignment
* Support for minor formatting differences and token stutters

These mechanisms reduce sensitivity to formatting artifacts while preserving validation of the underlying path transitions.

---

# QUICK START

## 1. Installation

Clone the workspace and install the required dependencies.

```bash
pip install -r requirements.txt
```

## 2. Execution

Run an evaluation using your preferred endpoint wrapper. One or more comma-separated gravity levels may be supplied.

```bash
python quickstart_api.py \
  --api_key "your_api_key" \
  --model "google/gemini-2.5-flash-lite" \
  --n_probes 3 \
  --gravity_levels "0.5,1.1,99.9"
```

## 3. Output

After execution, the benchmark aggregates the collected metrics and generates a summary report.

```
======================================================================
 FINAL GRAVITY SPECTRA METRIC SUMMARY
======================================================================
Gravity Horizon      | Fracture Depth (z-value)  | Volatility Score
----------------------------------------------------------------------
Gravity 0.5          | z = 9                     | V = 14.2%
Gravity 1.1          | z = 9                     | V = 11.5%
Gravity 99.9         | z = 9                     | V = 15.0%
======================================================================
SUCCESS! Results saved to outputs/benchmark_results.json
```

---

# FILE MANIFEST

| File                  | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `quickstart_api.py`   | CLI parsing, payload mapping, and reporting orchestration     |
| `benchmark_runner.py` | Adaptive search execution and path continuity validation      |
| `model_evaluation.py` | API client management and local vLLM inference                |
| `fracture_finder.py`  | Telemetry analysis and Cognitive Volatility Score calculation |
| `requirements.txt`    | Project dependencies                                          |
