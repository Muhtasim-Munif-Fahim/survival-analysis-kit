"""survival_kit: a lightweight survival analysis toolkit for right-censored data."""

from .concordance import ConcordanceResult, concordance_index
from .cox import CoxPHResult, fit_cox_ph
from .kaplan_meier import SurvivalCurve, fit_kaplan_meier
from .logrank import LogRankResult, log_rank_test, log_rank_test_groups
from .nelson_aalen import CumulativeHazard, fit_nelson_aalen
from .report import CohortSummary, format_p_value, render_report
from .rmst import (
    RMSTDifferenceResult,
    RMSTResult,
    default_truncation_time,
    restricted_mean_survival_time,
    rmst_difference_test,
)
from .synth import (
    PHSurvivalData,
    SurvivalData,
    generate_ph_data,
    generate_survival_data,
    load_csv,
    save_csv,
)

__version__ = "0.9.0"

__all__ = [
    "CohortSummary",
    "ConcordanceResult",
    "CoxPHResult",
    "CumulativeHazard",
    "LogRankResult",
    "PHSurvivalData",
    "RMSTDifferenceResult",
    "RMSTResult",
    "SurvivalCurve",
    "SurvivalData",
    "concordance_index",
    "default_truncation_time",
    "fit_cox_ph",
    "fit_kaplan_meier",
    "fit_nelson_aalen",
    "format_p_value",
    "generate_ph_data",
    "generate_survival_data",
    "load_csv",
    "log_rank_test",
    "log_rank_test_groups",
    "render_report",
    "restricted_mean_survival_time",
    "rmst_difference_test",
    "save_csv",
    "__version__",
]
