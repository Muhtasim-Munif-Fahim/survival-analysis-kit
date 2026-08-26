"""survival_kit: a lightweight survival analysis toolkit for right-censored data."""

from .kaplan_meier import SurvivalCurve, fit_kaplan_meier

__version__ = "0.2.0"

__all__ = ["SurvivalCurve", "fit_kaplan_meier", "__version__"]