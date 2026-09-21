"""Console UX for the ARCUS-X benchmark runner.

Provides three pieces of user-facing output that sit on top of the existing
``logging``-based diagnostics:

* :func:`print_banner`  - the startup banner (version, self-test result,
  loading messages).
* :class:`ProgressDisplay` - a live progress bar plus a transient status line
  (probe counter, percentage bar, current model/tier/gravity/horizon context,
  and status messages such as "Waiting for API..." or rate-limit retries).
* :func:`print_metrics_chart` - the final evaluation chart summarising the
  headline trajectory metrics, the CRI components, per-tier fracture depth and
  the full error taxonomy distribution.

The progress display writes to ``stdout`` and uses ANSI cursor control when the
stream is a TTY; when stdout is piped/redirected it falls back to throttled
plain lines so output is still useful and never corrupts the terminal.
"""
# If you're reading this, have a great day! :)
# The code is deterministic. The debugging process was not.
from __future__ import annotations

import shutil
import sys
from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - import shim
    from arcus import __version__ as ARCUS_VERSION
except Exception:  # pragma: no cover
    ARCUS_VERSION = "0.x"


def _term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:  # pragma: no cover
        return default


def _bar_chars(stream) -> tuple:
    """Return (full, empty) bar glyphs supported by ``stream``'s encoding.

    Uses Unicode block characters on UTF-8 capable streams and falls back to
    ASCII (``#`` / ``-``) on legacy code pages (e.g. Windows cp1252) where the
    block glyphs cannot be encoded.
    """
    enc = getattr(stream, "encoding", None) or ""
    try:
        "█░".encode(enc)
        return "█", "░"
    except Exception:  # pragma: no cover - encoding cannot represent blocks
        return "#", "-"


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
def print_banner(self_test_passed: bool, model_name: str = "") -> None:
    """Print the ARCUS-X startup banner to stdout.

    Parameters
    ----------
    self_test_passed:
        Whether the integrity / metric self-tests passed (drives the
        ``Self-test ... PASS/FAIL`` line).
    model_name:
        Optional model identifier echoed for context.
    """
    status = "PASS" if self_test_passed else "FAIL"
    lines = [
        f"ARCUS-X v{ARCUS_VERSION}",
        "Self-test " + "." * 10 + f" {status}",
        "Loading benchmark...",
        "Loading models......",
        "Starting evaluation.",
    ]
    if model_name:
        lines.append(f"Model: {model_name}")
    print("\n".join(lines))
    print()


# ---------------------------------------------------------------------------
# Live progress display
# ---------------------------------------------------------------------------
class ProgressDisplay:
    """Live progress display driven by adaptive search state."""

    BAR_WIDTH = 20

    def __init__(self, model_name: str = "", total: Optional[int] = None, stream=None):
        self.model_name = model_name
        self.completed_evaluations = 0
        self.active_frontier = 0
        self.search_depth = 0
        self.seed = None
        self.tier = None
        self.gravity = None
        self.horizon = None
        self.status = ""
        self.stream = stream or sys.stdout
        self.use_tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self._rendered_lines = 0
        self._last_status_printed = ""
        # Parallel / resume context
        self.seeds_done = 0
        self.seeds_total = 0
        self.workers_active = 0
        self.resume_loaded = 0
        self.resume_remaining = 0
        self.resume_mode = False

    # -- public API -----------------------------------------------------
    def set_context(self, tier: Any, gravity: Any, horizon: Any, seed: Any = None) -> None:
        """Update current active evaluation context."""
        self.seed = seed
        self.tier = tier
        self.gravity = gravity
        self.horizon = horizon
        if self.use_tty:
            self._render()

    def set_adaptive_state(self, completed_evaluations: Optional[int] = None, active_frontier: Optional[int] = None, search_depth: Optional[int] = None) -> None:
        """Update adaptive search state parameters."""
        if completed_evaluations is not None:
            self.completed_evaluations = int(completed_evaluations)
        if active_frontier is not None:
            self.active_frontier = int(active_frontier)
        if search_depth is not None:
            self.search_depth = int(search_depth)
        if self.use_tty:
            self._render()

    def set_status(self, status: str) -> None:
        """Set or clear the transient status line."""
        self.status = status or ""
        if self.status and ("waiting" in self.status.lower() or "computing" in self.status.lower() or "saving" in self.status.lower() or "completed" in self.status.lower()):
            self.seed = None
            self.tier = None
            self.gravity = None
            self.horizon = None
        if self.use_tty:
            self._render()
        else:
            if (self.status and self.status != self._last_status_printed
                    and self.status != "Waiting for API..."):
                self.stream.write(f"[status] {self.status}\n")
                self.stream.flush()
                self._last_status_printed = self.status
            elif not self.status:
                self._last_status_printed = ""

    def set_seed_progress(self, done: int, total: int, workers: int) -> None:
        """Update parallel seed/worker progress counters."""
        self.seeds_done = done
        self.seeds_total = total
        self.workers_active = workers
        if self.use_tty:
            self._render()
        else:
            self.stream.write(
                f"[seeds] {done}/{total} complete, {workers} worker(s) active\n"
            )
            self.stream.flush()

    def set_resume_info(self, loaded: int, remaining: int) -> None:
        """Enable resume-mode display with loaded/remaining adaptive counts."""
        self.resume_mode = True
        self.resume_loaded = loaded
        self.resume_remaining = remaining
        self.completed_evaluations = loaded
        self.active_frontier = remaining
        if self.use_tty:
            self._render()
        else:
            self.stream.write(
                f"[resume] Loaded {loaded} completed evaluations, "
                f"{remaining} remaining frontier. Skipping completed work...\n"
            )
            self.stream.flush()

    def increment(self, n: int = 1) -> None:
        """Advance completed evaluations counter and re-render."""
        self.completed_evaluations += n
        if self.active_frontier > 0:
            self.active_frontier = max(0, self.active_frontier - n)
        if self.use_tty:
            self._render()
        else:
            self._maybe_print_plain()

    def finish(self) -> None:
        """Tear down the live block (TTY) so subsequent output is clean."""
        if self.use_tty and self._rendered_lines:
            self.stream.write("\x1b[" + str(self._rendered_lines) + "A")
            self.stream.write("\x1b[J")
            self.stream.flush()
            self._rendered_lines = 0
        self.status = ""

    # -- rendering ------------------------------------------------------
    def _build(self) -> str:
        lines = [
            "ARCUS-X Execution State",
            "-----------------------",
            f"Completed evaluations: {self.completed_evaluations}",
            f"Active frontier: {self.active_frontier}",
            f"Search depth: {self.search_depth}",
            "",
            "Current evaluation:",
        ]
        if self.seed is not None and self.tier is not None and self.gravity is not None and self.horizon is not None:
            lines.extend([
                f"  Seed: {self.seed}",
                f"  Tier: {self.tier}",
                f"  Gravity: {self.gravity}",
                f"  Horizon: {self.horizon}",
            ])
        else:
            lines.append("  Waiting for adaptive search expansion...")

        if self.seeds_total > 0:
            seed_bar_width = self.BAR_WIDTH
            filled = int(round(self.seeds_done / max(1, self.seeds_total) * seed_bar_width))
            filled = max(0, min(seed_bar_width, filled))
            full, empty = _bar_chars(self.stream)
            seed_bar = full * filled + empty * (seed_bar_width - filled)
            lines.append("")
            lines.append(
                f"Seeds: [{seed_bar}] {self.seeds_done}/{self.seeds_total} complete"
            )
            lines.append(f"Workers: {self.workers_active} active")

        if self.resume_mode:
            if self.resume_remaining == 0:
                lines.append("")
                lines.append("ARCUS-X Resume")
                lines.append("----------------")
                lines.append(f"Completed evaluations: {self.completed_evaluations}")
                lines.append("")
                lines.append("✓ All evaluations already completed")
                lines.append("✓ No evaluation required")
            else:
                lines.append("")
                lines.append(f"Loaded: {self.resume_loaded} completed evaluations")
                lines.append(f"Remaining frontier: {self.active_frontier}")
                lines.append("Skipping completed work...")

        if self.status:
            lines.append("")
            lines.append(self.status)
        return "\n".join(lines) + "\n"

    def _render(self) -> None:
        if self._rendered_lines:
            self.stream.write("\x1b[" + str(self._rendered_lines) + "A")
            self.stream.write("\x1b[J")
        block = self._build()
        self.stream.write(block)
        self.stream.flush()
        self._rendered_lines = block.count("\n")

    def _maybe_print_plain(self) -> None:
        if self.seed is not None:
            self.stream.write(
                f"Completed evaluations: {self.completed_evaluations} | "
                f"seed={self.seed} tier={self.tier} gravity={self.gravity} horizon={self.horizon}\n"
            )
            self.stream.flush()


# ---------------------------------------------------------------------------
# Final metrics chart
# ---------------------------------------------------------------------------
def _hbar(value: float, max_value: float, width: int = 30, stream=None) -> str:
    if stream is None:
        stream = sys.stdout
    full, empty = _bar_chars(stream)
    if max_value <= 0:
        return empty * width
    ratio = max(0.0, min(1.0, float(value) / float(max_value)))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return full * filled + empty * (width - filled)


def _max_fracture(per_tier: Dict[str, Any]) -> float:
    vals = [float(d.get("fracture_depth", 0.0)) for d in per_tier.values()]
    return max(vals) if vals else 1.0


def print_metrics_chart(aggregated: Dict[str, Any]) -> None:
    """Render the final evaluation chart to stdout.

    Shows the headline trajectory metrics (0-100 bars), the secondary CRI
    components, fracture depth, per-tier fracture depth and the full error
    taxonomy distribution.
    """
    print()
    print("=" * 64)
    print("ARCUS-X EVALUATION CHART")
    print("=" * 64)

    headline = [
        ("Step Accuracy", aggregated.get("step_accuracy", 0.0)),
        ("Exact Match", aggregated.get("exact_match", 0.0)),
        ("Continuity", aggregated.get("continuity", 0.0)),
        ("Horizon Compliance", aggregated.get("horizon_compliance", 0.0)),
        ("Efficiency", aggregated.get("efficiency", 0.0)),
    ]
    print("\nHeadline Metrics:")
    for name, val in headline:
        print(f"  {name:<20} {_hbar(val, 100):<32} {float(val):>6.1f}%")

    print(f"\n  {'GBI':<20} {float(aggregated.get('generation_bloat_index', 0.0)):>6.2f}  (lower is better)")
    cri = aggregated.get("cri", {})
    if isinstance(cri, dict):
        print(f"  {'CRI':<20} {float(cri.get('cri', 0.0)):>6.3f}")
        print(f"  {'Trajectory Fidelity':<20} {float(cri.get('trajectory_fidelity', 0.0)):>6.3f}")
        print(f"  {'Horizon Robustness':<20} {float(cri.get('horizon_robustness', 0.0)):>6.3f}")
        print(f"  {'Semantic Robustness':<20} {float(cri.get('semantic_robustness', 0.0)):>6.3f}")
        print(f"  {'Generation Efficiency':<20} {float(cri.get('generation_efficiency', 0.0)):>6.3f}")

    fd = aggregated.get("fracture_depth", 0)
    print(f"\n  Fracture Depth: {fd}")

    per_tier = aggregated.get("per_tier", {})
    if per_tier:
        print("\nPer-Tier Fracture Depth:")
        max_fd = _max_fracture(per_tier)
        for tier, data in sorted(per_tier.items()):
            fdv = float(data.get("fracture_depth", 0.0))
            print(f"  {tier:<20} {_hbar(fdv, max(1.0, max_fd)):<32} {fdv:>6.1f}")

    fa = aggregated.get("failure_analysis", {})
    if fa:
        print("\nError Taxonomy (failure distribution):")
        for mode, pct in fa.items():
            print(f"  {mode:<30} {_hbar(float(pct), 100):<32} {float(pct):>6.1f}%")

    print("=" * 64)
