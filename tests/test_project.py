"""Tests for project save/load including the spin-adduct mixture."""
from epr_simfit import project


def test_mixture_and_conditions_roundtrip():
    mixture = {
        "component_ids": ["pbn_oh", "pbn_ch3_dmso"],
        "ratios": {"pbn_oh": 60.0, "pbn_ch3_dmso": 40.0},
        "fit_conditions": {"refine": ["linewidths", "g", "hyperfine"],
                           "fit_method": "Physics + Monte-Carlo"},
    }
    data = project.save_project(microwave_frequency_GHz=9.82, spectrum_text="1 2\n3 4\n",
                                mixture=mixture)
    obj = project.load_project(data)
    assert obj["mixture"]["component_ids"] == ["pbn_oh", "pbn_ch3_dmso"]
    assert obj["mixture"]["ratios"]["pbn_oh"] == 60.0
    assert obj["mixture"]["fit_conditions"]["fit_method"] == "Physics + Monte-Carlo"
    assert "linewidths" in obj["mixture"]["fit_conditions"]["refine"]


def test_summary_mentions_mixture():
    data = project.save_project(microwave_frequency_GHz=9.5,
                                mixture={"component_ids": ["dmpo_oh"], "ratios": {"dmpo_oh": 100.0}})
    assert "mixture: 1 adduct" in project.project_summary(project.load_project(data))


def test_empty_mixture_is_fine():
    data = project.save_project(microwave_frequency_GHz=9.5)
    obj = project.load_project(data)
    assert obj["mixture"] == {}
