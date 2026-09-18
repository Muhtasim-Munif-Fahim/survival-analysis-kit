"""survival_kit: a lightweight survival analysis toolkit for right-censored data."""

from .competing_risks import (
    CumulativeIncidence,
    GrayTestResult,
    fit_all_cumulative_incidence,
    fit_cumulative_incidence,
    gray_test,
    gray_test_groups,
)
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
    CompetingRisksData,
    PHSurvivalData,
    SurvivalData,
    exponential_competing_cif,
    generate_competing_risks_data,
    generate_ph_data,
    generate_survival_data,
    load_csv,
    save_csv,
)
from .utils import observed_causes

__version__ = "0.9.0"

__all__ = [
    "CohortSummary",
    "CompetingRisksData",
    "ConcordanceResult",
    "CoxPHResult",
    "CumulativeHazard",
    "CumulativeIncidence",
    "GrayTestResult",
    "LogRankResult",
    "PHSurvivalData",
    "RMSTDifferenceResult",
    "RMSTResult",
    "SurvivalCurve",
    "SurvivalData",
    "concordance_index",
    "default_truncation_time",
    "exponential_competing_cif",
    "fit_all_cumulative_incidence",
    "fit_cox_ph",
    "fit_cumulative_incidence",
    "fit_kaplan_meier",
    "fit_nelson_aalen",
    "format_p_value",
    "generate_competing_risks_data",
    "generate_ph_data",
    "generate_survival_data",
    "gray_test",
    "gray_test_groups",
    "load_csv",
    "log_rank_test",
    "log_rank_test_groups",
    "observed_causes",
    "render_report",
    "restricted_mean_survival_time",
    "rmst_difference_test",
    "save_csv",
    "__version__",
]
