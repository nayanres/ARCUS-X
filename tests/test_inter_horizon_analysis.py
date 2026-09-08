import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from inter_horizon_analysis import RawEntry, analyze_entries, render_report, split_horizon


def test_split_horizon_assigns_remainder_to_end():
    expected = {
        3: (1, 1, 1), 4: (1, 1, 2), 5: (1, 1, 3),
        6: (2, 2, 2), 7: (2, 2, 3), 8: (2, 2, 4),
        9: (3, 3, 3), 10: (3, 3, 4), 11: (3, 3, 5),
    }
    for z, sizes in expected.items():
        assert tuple(split_horizon(z).values()) == sizes


def _entry(z=3, output="[0,0]->[1,0]->[2,0]->[3,0]"):
    return RawEntry(
        z=z,
        grid="4x4",
        actions=["east"] * z,
        transition_rules={"east": {"dx": 1, "dy": 0}},
        raw_output=output,
        ground_truth="['[0,0]', '[1,0]', '[2,0]', '[3,0]']",
    )


def test_invalid_output_is_not_a_cognitive_error():
    item = analyze_entries([_entry(output="[EXCEPTION] provider failure")])[0]
    assert item.valid is False
    assert "State Tracking Failure: 0.0%" not in render_report([item])


def test_missing_steps_do_not_divide_by_zero():
    report = render_report(analyze_entries([_entry(output="[0,0]")]))
    assert "early:  N/A" in report
    assert "middle: N/A" in report
    assert "end:    N/A" in report


def test_all_tax_only_adds_detail_and_horizons_are_numeric():
    entries = [_entry(10, "[0,0]" + "->" + "->".join("[0,0]" for _ in range(10))),
               _entry(3)]
    default = render_report(analyze_entries(entries))
    detailed = render_report(analyze_entries(entries), all_tax=True)
    assert default == detailed.split("=" * 60 + "\nTaxonomy By Horizon And Trajectory Third", 1)[0].rstrip()
    assert detailed.index("z=3") < detailed.index("z=10")
