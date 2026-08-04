"""Physical constants and project-wide defaults."""

MU_B_OVER_H_GHZ_PER_T = 13.99624555
DEFAULT_MW_FREQUENCY_GHZ = 9.85
DEFAULT_FIELD_RANGE_MT = (330.0, 370.0)
EPS = 1e-12

DELTA_BIC_LABELS = [
    (2.0, "no meaningful improvement"),
    (6.0, "weak support"),
    (10.0, "moderate support"),
    (float("inf"), "strong statistical improvement"),
]


def delta_bic_label(delta_bic: float) -> str:
    """Return the evidence label associated with a Delta_BIC value."""
    for cutoff, label in DELTA_BIC_LABELS:
        if delta_bic < cutoff:
            return label
    return DELTA_BIC_LABELS[-1][1]
