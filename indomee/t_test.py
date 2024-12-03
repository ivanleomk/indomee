from itertools import combinations
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


def perform_t_tests(
    *populations, names: Optional[List[str]] = None, paired: bool = True
) -> pd.DataFrame:
    """Perform pairwise t-tests between multiple populations.

    Args:
        *populations: Variable number of populations to compare. Each population should be a list/array of numbers.
        names: Optional list of names for each population. If not provided, will use ["pop1", "pop2", etc.]
        paired: Whether to perform paired t-tests (default True). Populations must be same length if True.

    Returns:
        pd.DataFrame: DataFrame containing pairwise t-test results with columns:
            - population_1: Name of first population
            - population_2: Name of second population
            - t_statistic: T-test statistic
            - p_value: P-value from t-test
            - significant: Boolean indicating if p < 0.05
    """
    # Input validation
    if len(populations) < 2:
        raise ValueError("At least two populations are required for t-tests")

    # Generate default names if not provided
    if names is None:
        names = [f"pop{i+1}" for i in range(len(populations))]
    elif len(names) != len(populations):
        raise ValueError("Number of names must match number of populations")

    # Convert all populations to numpy arrays
    populations = [np.array(pop) for pop in populations]

    # Check lengths if paired
    if paired:
        lengths = [len(pop) for pop in populations]
        if len(set(lengths)) > 1:
            raise ValueError("All populations must be same length for paired t-tests")

    # Perform pairwise t-tests
    results = []
    for pop1_idx, pop2_idx in combinations(range(len(populations)), 2):
        pop1, pop2 = populations[pop1_idx], populations[pop2_idx]
        name1, name2 = names[pop1_idx], names[pop2_idx]

        if paired:
            t_stat, p_val = stats.ttest_rel(pop1, pop2)
        else:
            t_stat, p_val = stats.ttest_ind(pop1, pop2)

        results.append(
            {
                "population_1": name1,
                "population_2": name2,
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < 0.05,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Round numeric columns
    df["t_statistic"] = df["t_statistic"].round(2)
    df["p_value"] = df["p_value"].map("{:.2e}".format)

    return df
