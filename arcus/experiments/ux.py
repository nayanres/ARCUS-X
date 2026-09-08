"""Presentation-only terminal UX for ARCUS-X.

The renderer deliberately knows nothing about probe generation, scoring, or
adaptive-search decisions.  ``ProgressDisplay`` is a small event/state
adapter: the benchmark may continue using its existing callbacks while the
theme renderer consumes only display state.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TextIO

try:  # pragma: no cover - import shim
    from arcus import __version__ as ARCUS_VERSION
except Exception:  # pragma: no cover
    ARCUS_VERSION = "0.x"


THEME_NAMES = ("midnight", "minimal", "terminal", "high_contrast", "neon")


@dataclass(frozen=True)
class Theme:
    """Presentation settings only; no benchmark behavior belongs here."""

    name: str
    accent: str
    emphasis: str
    muted: str
    good: str
    warning: str
    border: str
    section: str
    symbols: Dict[str, str] = field(default_factory=dict)
    compact: bool = False


THEMES = {
    "midnight": Theme(
        "midnight", "36", "97", "90", "32", "33", "34", "36",
        {"corner": "╭", "corner_end": "╮", "bottom": "╰", "bottom_end": "╯",
         "tee": "├", "line": "─", "dot": "●", "ok": "✓", "arrow": "→"},
    ),
    "minimal": Theme(
        "minimal", "37", "97", "90", "37", "37", "37", "37",
        {"corner": "", "corner_end": "", "bottom": "", "bottom_end": "",
         "tee": "", "line": "-", "dot": "*", "ok": "+", "arrow": "->"},
        compact=True,
    ),
    "terminal": Theme(
        "terminal", "32", "37", "90", "32", "33", "37", "32",
        {"corner": "+", "corner_end": "+", "bottom": "+", "bottom_end": "+",
         "tee": "+", "line": "-", "dot": "*", "ok": "+", "arrow": "->"},
    ),
    "high_contrast": Theme(
        "high_contrast", "96", "97", "37", "92", "93", "97", "96",
        {"corner": "╔", "corner_end": "╗", "bottom": "╚", "bottom_end": "╝",
         "tee": "╠", "line": "═", "dot": "●", "ok": "✓", "arrow": "=>"},
    ),
    "neon": Theme(
        "neon", "95", "97", "90", "92", "93", "35", "95",
        {"corner": "╭", "corner_end": "╮", "bottom": "╰", "bottom_end": "╯",
         "tee": "├", "line": "━", "dot": "●", "ok": "✓", "arrow": "➜"},
    ),
}


def available_themes() -> Sequence[str]:
    return THEME_NAMES


def get_theme(name: str = "midnight") -> Theme:
    """Resolve a local preset, raising a useful CLI-compatible error."""
    if name not in THEMES:
        raise ValueError(
            f"Unknown theme: {name}\n"
            "Available themes: " + ", ".join(THEME_NAMES)
        )
    return THEMES[name]


def _term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except OSError:  # pragma: no cover
        return default


def _unicode_supported(stream: TextIO) -> bool:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "╭●✓".encode(encoding)
        return True
    except (LookupError, UnicodeError):
        return False


def _capabilities(stream: TextIO) -> Dict[str, bool]:
    interactive = bool(getattr(stream, "isatty", lambda: False)())
    encoding = (getattr(stream, "encoding", None) or "").lower()
    unicode = _unicode_supported(stream)
    ansi = interactive and not bool(__import__("os").environ.get("NO_COLOR"))
    if sys.platform == "win32" and interactive:
        ansi = ansi and bool(
            __import__("os").environ.get("WT_SESSION")
            or __import__("os").environ.get("TERM")
            or __import__("os").environ.get("ANSICON")
            or "utf" in encoding
        )
    return {"interactive": interactive, "ansi": ansi, "unicode": unicode}


def _bar_chars(stream) -> tuple:
    """Compatibility helper retained for callers of the old UX module."""
    try:
        "█░".encode(getattr(stream, "encoding", None) or "")
        return "█", "░"
    except (LookupError, UnicodeError):
        return "#", "-"


def print_banner(self_test_passed: bool, model_name: str = "") -> None:
    """Print the short startup message without affecting execution."""
    status = "PASS" if self_test_passed else "FAIL"
    lines = [
        f"ARCUS-X v{ARCUS_VERSION}",
        f"Self-test .......... {status}",
        "Loading benchmark...",
        "Loading models......",
        "Starting evaluation.",
    ]
    if model_name:
        lines.append(f"Model: {model_name}")
    print("\n".join(lines))
    print()


class ProgressDisplay:
    """Persistent presentation state updated by benchmark events."""

    def __init__(
        self,
        model_name: str = "",
        total: Optional[int] = None,
        stream: Optional[TextIO] = None,
        theme: str | Theme = "midnight",
    ):
        self.model_name = model_name
        self.total = total
        self.theme = get_theme(theme) if isinstance(theme, str) else theme
        self.stream = stream or sys.stdout
        self.capabilities = _capabilities(self.stream)
        self.use_tty = self.capabilities["interactive"] and self.capabilities["ansi"]
        self.completed_evaluations = 0
        self.active_frontier = 0  # compatibility/accounting field, not rendered
        self.search_depth = 0  # compatibility field, intentionally not rendered
        self.seed = self.tier = self.gravity = self.horizon = self.n_probe = None
        self.status = "Waiting for adaptive search expansion..."
        self.afs_status = "waiting"
        self.valid_count = self.invalid_count = self.cached_count = None
        self.path_history: List[Any] = []
        self.recent_probes: List[Dict[str, Any]] = []
        self.recent_decisions: List[str] = []
        self.last_decision: Optional[str] = None
        self.lookahead: Optional[Any] = None
        self.seeds_done = self.seeds_total = self.workers_active = 0
        self.resume_loaded = self.resume_remaining = 0
        self.resume_mode = False
        self._rendered_lines = 0
        self._last_plain_context = None
        self._pulse = False

    def _refresh(self) -> None:
        if self.use_tty:
            self._render()
        elif self.status and self.status != self._last_plain_context:
            self._maybe_print_plain()

    def set_context(self, tier: Any, gravity: Any, horizon: Any, seed: Any = None, n_probe: Any = None) -> None:
        self.seed, self.tier, self.gravity, self.horizon = seed, tier, gravity, horizon
        if n_probe is not None:
            self.n_probe = n_probe
        if horizon is not None and (not self.path_history or self.path_history[-1] != horizon):
            self.path_history.append(horizon)
        self.afs_status = "active"
        self.status = "Adaptive search active"
        self._refresh()

    def set_adaptive_state(
        self,
        completed_evaluations: Optional[int] = None,
        active_frontier: Optional[int] = None,
        search_depth: Optional[int] = None,
    ) -> None:
        if completed_evaluations is not None:
            self.completed_evaluations = int(completed_evaluations)
        if active_frontier is not None:
            self.active_frontier = int(active_frontier)
        if search_depth is not None:
            self.search_depth = int(search_depth)
        self._refresh()

    def set_status(self, status: str) -> None:
        self.status = status or ""
        lowered = self.status.lower()
        if "completed" in lowered or "no evaluation" in lowered:
            self.afs_status = "completed"
        elif "waiting" in lowered:
            self.afs_status = "waiting"
        elif status:
            self.afs_status = "active"
        self._refresh()

    def set_seed_progress(self, done: int, total: int, workers: int) -> None:
        self.seeds_done, self.seeds_total, self.workers_active = done, total, workers
        self._refresh()

    def set_resume_info(self, loaded: int, remaining: int) -> None:
        self.resume_mode, self.resume_loaded, self.resume_remaining = True, loaded, remaining
        self.completed_evaluations, self.active_frontier = loaded, remaining
        self.status = "All evaluations already completed" if remaining == 0 else "Resuming saved work"
        self.afs_status = "completed" if remaining == 0 else "active"
        self._refresh()

    def record_probe(
        self, horizon: Any, accuracy: Any = None, valid: Optional[bool] = None,
        cached: bool = False, result: Optional[str] = None,
    ) -> None:
        """Accept optional probe telemetry; it cannot influence benchmark state."""
        item = {"horizon": horizon, "accuracy": accuracy, "valid": valid, "cached": cached,
                "result": result or ("OK" if valid is not False else "INVALID")}
        self.recent_probes.append(item)
        self.recent_probes = self.recent_probes[-7:]
        if valid is True:
            self.valid_count = (self.valid_count or 0) + 1
        elif valid is False:
            self.invalid_count = (self.invalid_count or 0) + 1
        if cached:
            self.cached_count = (self.cached_count or 0) + 1
        self._refresh()

    def record_decision(self, source: Any, target: Any, decision: str, lookahead: Any = None) -> None:
        self.last_decision = f"{source} {self._symbol('arrow')} {target}"
        self.lookahead = lookahead
        self.recent_decisions.append(f"{self.last_decision}  {decision}")
        self.recent_decisions = self.recent_decisions[-3:]
        self._refresh()

    def set_metrics(self, valid: Optional[int] = None, invalid: Optional[int] = None,
                    cached: Optional[int] = None) -> None:
        self.valid_count, self.invalid_count, self.cached_count = valid, invalid, cached
        self._refresh()

    def increment(self, n: int = 1) -> None:
        self.completed_evaluations += n
        if self.active_frontier > 0:
            self.active_frontier = max(0, self.active_frontier - n)
        self._refresh()

    def finish(self) -> None:
        if self.use_tty and self._rendered_lines:
            self.stream.write(f"\x1b[{self._rendered_lines}A\x1b[J")
            self.stream.flush()
        self._rendered_lines = 0

    def _symbol(self, name: str) -> str:
        symbols = self.theme.symbols
        if not self.capabilities["unicode"]:
            return {"corner": "+", "corner_end": "+", "bottom": "+", "bottom_end": "+",
                    "tee": "+", "line": "-", "dot": "*", "ok": "+", "arrow": "->"}.get(name, "")
        return symbols.get(name, "")

    def _paint(self, text: str, code: str) -> str:
        if not self.capabilities["ansi"]:
            return text
        return f"\033[{code}m{text}\033[0m"

    def _label(self, label: str, value: Any) -> str:
        return f"  {label}: {value}"

    def _section(self, title: str, width: int) -> str:
        line = self._symbol("line") * max(1, width - len(title) - 5)
        if self.theme.compact:
            return f"-- {title} {line}"
        return f"{self._symbol('tee')}{self._symbol('line')} {title} {line}"

    def _build(self) -> str:
        width = max(58, min(_term_width(), 100)) - 2
        dot = self._symbol("dot") if self._pulse else (self._symbol("ok") if self.afs_status == "completed" else self._symbol("dot"))
        model = self.model_name or "-"
        tier = self.tier if self.tier is not None else "-"
        lines = [
            f"{self._symbol('corner')} ARCUS-X {' ' * max(1, width - 12)}{self._symbol('corner_end')}",
            self._label("MODEL", model),
            self._label("STATUS", f"{dot} {self.afs_status.upper()}"),
            self._section("SEARCH", width),
        ]
        if self.tier is not None:
            lines.append(self._label("Tier", tier))
        if self.gravity is not None:
            lines.append(self._label("Gravity", self.gravity))
        if self.seed is not None:
            lines.append(self._label("Seed", self.seed))
        if self.n_probe is not None:
            lines.append(self._label("N-Probe", self.n_probe))
        if self.horizon is not None:
            lines.append(self._label("Depth", self.horizon))
        lines.extend([
            "",
            "  CURRENT PATH",
            "  " + (f" {self._symbol('line')} ".join(str(x) for x in self.path_history[-10:]) or
                   "Waiting for adaptive search expansion..."),
        ])
        if self.last_decision:
            lines.append(self._label("Last decision", self.last_decision))
        if self.recent_decisions:
            lines.append(self._label("Decision", self.recent_decisions[-1].split("  ")[-1]))
        if self.lookahead is not None:
            lines.append(self._label("Look-ahead", f"{self.lookahead} horizons"))
        lines.extend([self._section("RECENT PROBES", width),
                      "  Horizon       Accuracy       Result"])
        for probe in self.recent_probes[-7:]:
            accuracy = "-" if probe["accuracy"] is None else f"{float(probe['accuracy']):.1f}%"
            result = self._symbol("ok") if probe["valid"] is not False else "!"
            lines.append(f"  z={str(probe['horizon']):<10} {accuracy:<14} {result}")
        if not self.recent_probes:
            lines.append("  No completed probes recorded yet")
        lines.extend([self._section("EXECUTION", width),
                      self._label("Completed", f"{self.completed_evaluations:,}"),
                      self._label("Valid", "-" if self.valid_count is None else f"{self.valid_count:,}"),
                      self._label("Invalid", "-" if self.invalid_count is None else f"{self.invalid_count:,}"),
                      self._label("Cached", "-" if self.cached_count is None else f"{self.cached_count:,}"),
                      "",
                      f"  {dot} {self.status or 'Idle'}"])
        if self.resume_mode:
            lines.extend([
                "  ARCUS-X Resume",
                self._label("Completed evaluations", f"{self.completed_evaluations:,}"),
            ])
            if self.resume_remaining == 0:
                lines.extend(["  " + self._symbol("ok") + " All evaluations already completed",
                              "  " + self._symbol("ok") + " No evaluation required"])
        if self.seeds_total:
            lines.append(self._label("Seeds", f"{self.seeds_done}/{self.seeds_total}  workers={self.workers_active}"))
        if self.resume_mode and self.resume_remaining:
            lines.append(self._label("Resume", f"{self.resume_loaded:,} loaded"))
        if self.theme.compact:
            return "\n".join(line.rstrip() for line in lines if line.strip()) + "\n"
        lines.append(f"{self._symbol('bottom')}{self._symbol('line') * width}{self._symbol('bottom_end')}")
        return "\n".join(lines) + "\n"

    def _render(self) -> None:
        if self._rendered_lines:
            self.stream.write(f"\x1b[{self._rendered_lines}A\x1b[J")
        self._pulse = not self._pulse
        block = self._build()
        self.stream.write(block)
        self.stream.flush()
        self._rendered_lines = block.count("\n")

    def _maybe_print_plain(self) -> None:
        self._pulse = not self._pulse
        block = self._build()
        if block != self._last_plain_context:
            self.stream.write(block)
            self.stream.flush()
            self._last_plain_context = block


def _hbar(value: float, max_value: float, width: int = 30, stream=None) -> str:
    stream = stream or sys.stdout
    full, empty = _bar_chars(stream)
    if max_value <= 0:
        return empty * width
    filled = max(0, min(width, int(round(max(0.0, min(1.0, float(value) / max_value)) * width))))
    return full * filled + empty * (width - filled)


def _max_fracture(per_tier: Dict[str, Any]) -> float:
    vals = [float(d.get("fracture_depth", 0.0)) for d in per_tier.values()]
    return max(vals) if vals else 1.0


def print_metrics_chart(aggregated: Dict[str, Any]) -> None:
    """Render final metrics; this output is intentionally separate from live UX."""
    print("\n" + "=" * 64 + "\nARCUS-X EVALUATION CHART\n" + "=" * 64)
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
        for name, key in (("CRI", "cri"), ("Trajectory Fidelity", "trajectory_fidelity"),
                          ("Horizon Robustness", "horizon_robustness"),
                          ("Semantic Robustness", "semantic_robustness"),
                          ("Generation Efficiency", "generation_efficiency")):
            print(f"  {name:<20} {float(cri.get(key, 0.0)):>6.3f}")
    print(f"\n  Fracture Depth: {aggregated.get('fracture_depth', 0)}")
    per_tier = aggregated.get("per_tier", {})
    if per_tier:
        print("\nPer-Tier Fracture Depth:")
        max_fd = _max_fracture(per_tier)
        for tier, data in sorted(per_tier.items()):
            value = float(data.get("fracture_depth", 0.0))
            print(f"  {tier:<20} {_hbar(value, max(1.0, max_fd)):<32} {value:>6.1f}")
    failure_analysis = aggregated.get("failure_analysis", {})
    if failure_analysis:
        print("\nError Taxonomy (failure distribution):")
        for mode, pct in failure_analysis.items():
            print(f"  {mode:<30} {_hbar(float(pct), 100):<32} {float(pct):>6.1f}%")
    print("=" * 64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preview an ARCUS-X local terminal theme.")
    parser.add_argument("--theme", default="midnight", choices=None)
    args = parser.parse_args(argv)
    try:
        theme = get_theme(args.theme)
    except ValueError as error:
        parser.exit(2, f"{error}\n")
    display = ProgressDisplay(
        model_name="GPT-5-mini", stream=sys.stdout, theme=theme
    )
    display.set_context(tier="T2", gravity=3.0, horizon=39, seed=219, n_probe=5)
    for horizon, accuracy in ((35, 70.0), (38, 70.0), (39, 70.0), (40, 70.0),
                              (50, 60.0), (52, 60.0), (53, 60.0)):
        display.record_probe(horizon, accuracy=accuracy, valid=True)
    display.record_decision(38, 39, "EXPAND", lookahead=3)
    display.set_metrics(valid=2310, invalid=111, cached=1942)
    display.set_status("Running probes...")
    if not display.use_tty:
        sys.stdout.write(display._build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
