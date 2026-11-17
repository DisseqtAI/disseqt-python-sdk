"""Rogan-Gladen prevalence correction: turn an observed judge pass rate into an estimate of the
true rate, using the judge's measured TPR and TNR."""

MIN_YOUDEN = 0.05


def corrected_rate(observed: float, tpr: float, tnr: float) -> float | None:
    """``None`` when the judge is too close to chance (Youden J <= 0.05) for a correction to mean
    anything; the estimate is clipped to [0, 1]."""
    j = tpr + tnr - 1.0
    if j <= MIN_YOUDEN:
        return None
    return min(1.0, max(0.0, (observed + tnr - 1.0) / j))
