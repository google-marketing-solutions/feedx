# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simulate valid designs and summarize results."""

from collections.abc import Collection
import dataclasses
import itertools
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas.io.formats import style
from scipy import stats

with warnings.catch_warnings():
  warnings.filterwarnings("ignore")
  import tqdm.autonotebook as tqdm

from feedx import data_preparation
from feedx import experiment_analysis
from feedx import experiment_design
from feedx import statistics

ExperimentDesign = experiment_design.ExperimentDesign


def generate_all_valid_designs(
    n_items_before_trimming: int,
    crossover_design_allowed: bool,
    traditional_design_allowed: bool,
    candidate_runtime_weeks: Collection[int],
    candidate_pretest_weeks: Collection[int],
    candidate_pre_trim_top_percentiles: Collection[float],
    candidate_pre_trim_bottom_percentiles: Collection[float],
    candidate_post_trim_percentiles: Collection[float],
    primary_metric: str,
    crossover_washout_weeks: int | None = None,
) -> list[ExperimentDesign]:
  """Generates valid experiment designs with user-defined parameters.

  This will generate all valid combinations of the candidate parameters.

  Args:
    n_items_before_trimming: The number of items in the data before trimming.
    crossover_design_allowed: Is a crossover experiment design allowed?
    traditional_design_allowed: Is a traditional experiment design allowed?
    candidate_runtime_weeks: The options for the number of weeks to run the
      experiment.
    candidate_pretest_weeks: The options for the number of weeks for data prior
      to the start of the experiment to use for trimming and cuped adjustment.
    candidate_pre_trim_top_percentiles: The options for the percetage of highest
      values of the primary metric to remove before the experiment begins, based
      on the pretest data.
    candidate_pre_trim_bottom_percentiles: The options for the percetage of
      lowest values of the primary metric to remove before the experiment
      begins, based on the pretest data.
    candidate_post_trim_percentiles: The options for the percetage of highest
      and lowest values of the primary metric to remove from the analysis after
      the experiment has concluded, based on the runtime data.
    primary_metric: The main ads performance metric to design the experiment
      for.
    crossover_washout_weeks: The number of weeks to exclude at the start of each
      crossover period, if using a crossover experiment. None if crossover is
      not allowed.

  Returns:
    List of valid experiment designs

  Raises:
    ValueError: If the crossover_washout_weeks are not specified but
      crossover_design_allowed is True.
  """
  if crossover_design_allowed & (crossover_washout_weeks is None):
    raise ValueError(
        "When allowing a crossover design you must specify the number of"
        " crossover washout weeks."
    )

  valid_designs = []
  design_constants = dict(
      n_items_before_trimming=n_items_before_trimming,
      primary_metric=primary_metric,
  )
  design_inputs = itertools.product(
      candidate_runtime_weeks,
      candidate_pretest_weeks,
      candidate_pre_trim_top_percentiles,
      candidate_pre_trim_bottom_percentiles,
      candidate_post_trim_percentiles,
  )

  for (
      runtime_weeks,
      pretest_weeks,
      pre_trim_top_percentile,
      pre_trim_bottom_percentile,
      post_trim_percentile,
  ) in design_inputs:
    has_pretrimming = (pre_trim_top_percentile + pre_trim_bottom_percentile) > 0
    if has_pretrimming & (pretest_weeks == 0):
      # Cannot do pretrimming without pretest weeks.
      continue

    if traditional_design_allowed:
      valid_designs.append(
          ExperimentDesign(
              runtime_weeks=runtime_weeks,
              pretest_weeks=pretest_weeks,
              pre_trim_top_percentile=pre_trim_top_percentile,
              pre_trim_bottom_percentile=pre_trim_bottom_percentile,
              post_trim_percentile=post_trim_percentile,
              is_crossover=False,
              **design_constants,
          )
      )

    can_crossover = (
        crossover_design_allowed
        & (runtime_weeks > 2 * crossover_washout_weeks)
        & (runtime_weeks % 2 == 0)
    )

    if can_crossover:
      valid_designs.append(
          ExperimentDesign(
              runtime_weeks=runtime_weeks,
              pretest_weeks=pretest_weeks,
              pre_trim_top_percentile=pre_trim_top_percentile,
              pre_trim_bottom_percentile=pre_trim_bottom_percentile,
              post_trim_percentile=post_trim_percentile,
              is_crossover=True,
              crossover_washout_weeks=crossover_washout_weeks,
              **design_constants,
          )
      )

  return valid_designs


def calculate_minimum_start_week_id(
    candidate_runtime_weeks: Collection[int],
    candidate_pretest_weeks: Collection[int],
    historical_week_ids: Collection[int],
) -> int:
  """Generate the earliest start week id to be used in the simulations.

  There must be enough week ids before the start week id to include the maximum
  pretest weeks, and enough week ids after the start week id for the maximum
  runtime.

  Args:
    candidate_runtime_weeks: Candidates for the number of weeks to run the
      experiment.
    candidate_pretest_weeks: Candidates for the number of weeks of data to use
      prior to running the experiment.
    historical_week_ids: All of the unique week ids available in the historical
      data.

  Returns:
    The minimum start week id, inclusive.

  Raises:
    ValueError: If the number of weeks of historical data is not sufficient to
      find a valid start date for those candidate runtime weeks and pretest
      weeks.
    ValueError: If any of the input collections are empty.
  """
  max_runtime_weeks = max(candidate_runtime_weeks)
  max_pretest_weeks = max(candidate_pretest_weeks)
  first_week_id = min(historical_week_ids)
  weeks_in_historical_data = max(historical_week_ids) - first_week_id + 1

  if weeks_in_historical_data < (max_runtime_weeks + max_pretest_weeks):
    raise ValueError(
        "There is not enough historical data to select a valid range of "
        f"start week ids: {weeks_in_historical_data=}, {max_runtime_weeks=}, "
        f"{max_pretest_weeks=}."
    )

  return first_week_id + max_pretest_weeks


def _bootstrap_sample_data(
    data: pd.DataFrame,
    rng: np.random.Generator,
    item_id_index_name: str,
    sample_weight_column: str = "n_samples_",
    dummy_list_column: str = "dummy_list_",
) -> pd.DataFrame:
  """Bootstrap sample the data.

  This creates a bootstrap sampled copy of the data, which is the equivalent
  of sampling the same number of rows as in the original set, with replacement.

  The item_id_index_name is overwritten with a new index, to ensure it is
  unique.

  Args:
    data: The dataset to sample.
    rng: A random number generator to perform the sampling.
    item_id_index_name: The name of the index level containing the item id.
    sample_weight_column: A column name used internally to do the sampling. It
      is not returned, and it should not already exist in the data.
    dummy_list_column: The column name used internally to do the sampling. It is
      not returned, and it should not already exist in the data.

  Returns:
    The sampled dataframe.

  Raises:
    ValueError: If the input data contains the sample_weight_column or
      dummy_list_column names.
  """
  protected_columns = {sample_weight_column, dummy_list_column}
  if protected_columns & set(data.columns):
    raise ValueError(
        "The input data must not contain the following columns: "
        f"{protected_columns & set(data.columns)}"
    )
  sampled_data = data.sample(frac=1.0, random_state=rng, replace=True)
  new_item_ids = np.arange(len(sampled_data.index.values))

  if isinstance(data.index, pd.MultiIndex):
    new_array_values = [
        new_item_ids
        if name == item_id_index_name
        else sampled_data.index.get_level_values(name)
        for name in sampled_data.index.names
    ]
    sampled_data.index = pd.MultiIndex.from_arrays(
        new_array_values, names=sampled_data.index.names
    )
  else:
    sampled_data.index = pd.Index(new_item_ids, name=item_id_index_name)

  return sampled_data


def apply_random_treatment_assignment(
    data: pd.DataFrame,
    *,
    rng: np.random.Generator,
    item_id_column: str,
    treatment_column: str = "treatment",
) -> pd.DataFrame:
  """Apply random treatment assignment to the data.

  This will independently randomise each item_id of the input dataset into
  treatment and control groups. It adds a column to the data, containing 1 if
  that row is in the treatment group, and 0 if that row is in the control group.

  Args:
    data: The data to randomise into treatment and control.
    rng: A random number generator to perform the randomisation.
    item_id_column: The column name of the item_id column (input).
    treatment_column: The column name of the treatment assignment column
      (output).

  Returns:
    The dataset with the "treatment" column added.

  Raises:
    ValueError: If the data already contains the treatment_column, or does
      not contain the item_id_column.
  """
  if treatment_column in data.columns:
    raise ValueError(
        "The data already contains a treatment column, cannot add one."
    )

  if item_id_column not in data.columns:
    raise ValueError("The data does not contain the item_id column.")

  items = data[[item_id_column]].drop_duplicates()
  items[treatment_column] = rng.choice(2, size=len(items))
  data = data.merge(items, on=item_id_column)
  return data


def _apply_synthetic_treatment_effect_to_regular_experiment(
    pivoted_data: pd.DataFrame,
    metric_name: str,
    effect_size: float,
    treatment_assignment_index_name: str,
) -> pd.DataFrame:
  is_treated = (
      pivoted_data.index.get_level_values(treatment_assignment_index_name) == 1
  )
  pivoted_data.loc[is_treated, (metric_name, "test")] += effect_size
  return pivoted_data


def _apply_synthetic_treatment_effect_to_crossover_experiment(
    pivoted_data: pd.DataFrame,
    metric_name: str,
    effect_size: float,
    treatment_assignment_index_name: str,
) -> pd.DataFrame:
  is_treated = (
      pivoted_data.index.get_level_values(treatment_assignment_index_name) == 1
  )
  pivoted_data.loc[is_treated, (metric_name, "test_1")] += effect_size
  pivoted_data.loc[~is_treated, (metric_name, "test_2")] += effect_size
  return pivoted_data


def apply_synthetic_treatment_effect(
    pivoted_data: pd.DataFrame,
    *,
    metric_name: str,
    design: ExperimentDesign,
    effect_size: float,
    treatment_assignment_index_name: str,
) -> pd.DataFrame:
  """Applies the synthetic treatment effect to the dataframe.

  If the design is a crossover design, then the pivoted_data should contain
  columns named "test_1" and "test_2" which contain the metric to add a
  synthetic treatment effect to. If the design is a regular design then the
  pivoted_data should contain a single column named "test" which contains the
  metric to apply the synthetic treatment effect to.

  The synthetic treatment will increase the metric by the effect_size for
  treatment when they have treatment applied. For a regular experiment, this
  is all the items where treatment = 1. For a crossover experiment, this
  is all the items where treatment = 1 for test_1, and all the columns
  where treatment_column = 0 for test_2.

  Args:
    pivoted_data: The dataframe to apply the synthetic treatment to. Must
      include the treatment_column, as well as "test" if it is not a crossover
      design, or "test_1" and "test_2" if it is a crossover design.
    metric_name: The metric to apply the synthetic treatment effect to.
    design: The experiment design being simulated, with the is_crossover
      attribute dictating whether this is a crossover design or regular design.
    effect_size: The effect size to apply to the metric when it is treated.
    treatment_assignment_index_name: The index level name that has the treatment
      status in the data.

  Returns:
    The dataframe with the synthetic treatment effect applied.

  Raises:
    ValueError: If the pivoted_data is missing any required columns.
  """
  if treatment_assignment_index_name not in pivoted_data.index.names:
    raise ValueError(
        "The pivoted_data does not contain the treatment_assignment_index_name."
    )

  if design.is_crossover:
    required_columns = {(metric_name, "test_1"), (metric_name, "test_2")}
  else:
    required_columns = {(metric_name, "test")}

  missing_columns = required_columns - set(pivoted_data.columns)
  if missing_columns:
    raise ValueError(
        "The pivoted_data is missing the following required columns: "
        f"{missing_columns}"
    )

  if design.is_crossover:
    return _apply_synthetic_treatment_effect_to_crossover_experiment(
        pivoted_data, metric_name, effect_size, treatment_assignment_index_name
    )
  else:
    return _apply_synthetic_treatment_effect_to_regular_experiment(
        pivoted_data, metric_name, effect_size, treatment_assignment_index_name
    )


@dataclasses.dataclass
class SimulationAnalysis:
  """A simultation analysis of the experiment design using historical data.

  This uses the historical data to estimate the minimum detectable effect
  measurable using this experiment design. Then optionally it can be used to
  run a validation simulation, which runs a variety of simulation tests to
  validate that the statistical tests are performing as expected when applied
  to real data.

  Attributes:
    design: The experiment design being analyzed.
    historical_data: The historical data used for the simulations.
    item_id_column: The column name that has the item ID in the historical data.
    week_id_column: The column name that has the week ID in the historical data.
    time_period_column: The column name that will be used internally to identify
      the time period. This column name must not already exist in the data.
      Defaults to "time_period".
    treatment_column: The column name that will be used internally to set the
      treatment assignment. This column name must not already exist in the data.
      Defaults to "treatment".
    minimum_start_week_id: The minimum start week ID that can be used for the
      simulations.
    maximum_start_week_id: The maximum start week ID that can be used for the
      simulations.
    valid_start_week_ids: The valid start week ids for this simulation.
    rng: The random number generator used for these simulations.
    minimum_detectable_effect: The smallest detectable improvement that can be
      reliably measured in this experiment. None if
      estimate_minimum_detectable_effect has not been run.
    relative_minimum_detectable_effect: The smallest detectable relative
      improvement that can be reliably measured in this experiment. None if
      estimate_minimum_detectable_effect has not been run.
    primary_metric_average: The average of the primary metric. None if
      estimate_minimum_detectable_effect has not been run.
    primary_metric_standard_deviation: The standard deviation of the primary
      metric. None if estimate_minimum_detectable_effect has not been run.
    null_p_value_robustness_p_value: A p-value testing the null hypothesis that
      the A/A simulation p-values follow a uniform distribution. A statistically
      significant p-value indicates an invalid experiment. None if the
      validation has not been run.
    false_positive_rate_robustness_p_value: A p-value testing the null
      hypothesis that the simulated false positive rate of this experiment is
      not more than the target design.alpha. A statistically significant p-value
      indicates an invalid experiment. None if the validation has not been run.
    power_robustness_p_value: A p-value testing the null hypothesis that the
      simulated power is at least the target power. A statistically significant
      p-value indicates an invalid experiment. None if the validation has not
      been run.
    aa_point_estimate_robustness_p_value: A p-value testing the null hypothesis
      that the point estimates of the A/A simulations are centered on 0.0. A
      statistically significant p-value indicates an invalid experiment. None if
      the validation has not been run.
    ab_point_estimate_robustness_p_value: A p-value testing the null hypothesis
      that the point estimates of the A/B simulations are centered on the
      minimum detectable effect. A statistically significant p-value indicates
      an invalid experiment. None if the validation has not been run.
    simulated_power_at_minimum_detectable_effect: The actual simulated power for
      the minimum detectable effect, should be close to design.power. None if
      the validation has not been run.
    simulated_false_positive_rate: The actual simulated false positive rate,
      should be close to design.alpha. None if the validation has not been run.
    aa_simulation_results: The full A/A simulation results. None if the
      validation has not been run.
    ab_simulation_results: The full A/B simulation results. None if the
      validation has not been run.
    robustness_p_value_threshold: The value to compare all the robustness
      p-values against to see if they pass. This should be very small, as
      normally the experiments will be valid, so there needs to be strong
      evidence to conclude otherwise. Defaults to 0.0001.
    null_p_value_robustness_check_pass: The null p_value robustness check
      passes, meaning the p_value is higher than the
      robustness_p_value_threshold. None if the validation has not been run.
    false_positive_rate_robustness_check_pass: The false positive rate
      robustness check passes, meaning the p_value is higher than the
      robustness_p_value_threshold. None if the validation has not been run.
    power_robustness_check_pass: The power robustness check passes, meaning the
      p_value is higher than the robustness_p_value_threshold. None if the
      validation has not been run.
    aa_point_estimate_robustness_check_pass: The A/A test point estimate
      robustness check passes, meaning the p_value is higher than the
      robustness_p_value_threshold. None if the validation has not been run.
    ab_point_estimate_robustness_check_pass: The A/B test point estimate
      robustness check passes, meaning the p_value is higher than the
      robustness_p_value_threshold. None if the validation has not been run.
    all_robustness_checks_pass: All the robustness checks pass, meaning all of
      the p-values are above the robustness_p_value_threshold. None if the
      validation has not been run.
  """

  design: ExperimentDesign = dataclasses.field(repr=False)
  historical_data: pd.DataFrame = dataclasses.field(repr=False)
  item_id_column: str = dataclasses.field(repr=False)
  week_id_column: str = dataclasses.field(repr=False)
  minimum_start_week_id: int = dataclasses.field(repr=False)
  maximum_start_week_id: int = dataclasses.field(repr=False, init=False)
  rng: np.random.Generator = dataclasses.field(repr=False)
  time_period_column: str = dataclasses.field(repr=False, default="time_period")
  treatment_column: str = dataclasses.field(repr=False, default="treatment")
  robustness_p_value_threshold: float = dataclasses.field(
      repr=False, default=0.0001
  )

  minimum_detectable_effect: float | None = dataclasses.field(
      default=None, init=False
  )
  relative_minimum_detectable_effect: float | None = dataclasses.field(
      default=None, init=False
  )
  primary_metric_average: float | None = dataclasses.field(
      default=None, init=False
  )
  primary_metric_standard_deviation: float | None = dataclasses.field(
      default=None, init=False
  )

  aa_simulation_results: pd.DataFrame | None = dataclasses.field(
      repr=False, default=None, init=False
  )
  ab_simulation_results: pd.DataFrame | None = dataclasses.field(
      repr=False, default=None, init=False
  )

  def __post_init__(self):
    maximum_week_id = self.historical_data[self.week_id_column].max()
    self.maximum_start_week_id = maximum_week_id - self.design.runtime_weeks + 1

  @property
  def valid_start_week_ids(self) -> range:
    return range(self.minimum_start_week_id, self.maximum_start_week_id + 1)

  def _get_pivoted_and_trimmed_historical_data(
      self, start_week_id: int, bootstrap_sample: bool = False
  ) -> pd.DataFrame:
    """Prepares the historical data for calculating the statistics.

    It pivots the dataset so that it has one item id per row, and then if
    required perfoms any trimming based on the pretest data.

    Bootstrap sampling can be applid, which is used for the simulations.

    Args:
      start_week_id: The start week id for this simulation.
      bootstrap_sample: Whether to bootstrap sample the historical data before
        estimating the stats. Defaults to false.

    Returns:
      The pivoted and trimmed historical data.

    Raises:
      RuntimeError: If the number of samples in the historical data does not
        match the number of samples in the design and it is not a bootstrap
        sample.
    """
    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.historical_data,
        design=self.design,
        start_week_id=start_week_id,
        metric_columns=[self.design.primary_metric],
        item_id_column=self.item_id_column,
        week_id_column=self.week_id_column,
        time_period_column=self.time_period_column,
    )

    if bootstrap_sample:
      pivoted_data = _bootstrap_sample_data(
          pivoted_data, self.rng, self.item_id_column
      )

    if (not bootstrap_sample) and (
        len(pivoted_data.index.values) != self.design.n_items_before_trimming
    ):
      raise RuntimeError(
          "Unexpected number of items before trimming. Design expects "
          f"{self.design.n_items_before_trimming:,}, but historical data has "
          f"{len(pivoted_data.index.values):,}"
      )

    if self.design.pretest_weeks > 0:
      pivoted_data = data_preparation.trim_outliers(
          pivoted_data,
          order_by=(self.design.primary_metric, "pretest"),
          trim_percentile_top=self.design.pre_trim_top_percentile,
          trim_percentile_bottom=self.design.pre_trim_bottom_percentile,
          rng=self.rng,
      )

    if (not bootstrap_sample) and (
        len(pivoted_data.index.values) != self.design.n_items_after_pre_trim
    ):
      raise RuntimeError(
          "Unexpected number of items after pre-trimming. Design expects "
          f"{self.design.n_items_after_pre_trim:,}, but historical data has "
          f"{len(pivoted_data.index.values):,}"
      )

    return pivoted_data

  def _estimate_primary_metric_stats(
      self, start_week_id: int
  ) -> tuple[float, float, int]:
    """Estimates the stats of the primary metric.

    This esimates the mean, standard deviation and sample size of the
    primary metric from the historical data, assuming the experiment starts
    on the given start week id.

    Args:
      start_week_id: The start week id for this simulation.

    Returns:
      The average, variance and sample size of the primary metric, for this
      start week id.

    Raises:
      RuntimeError: If the sample size does not match the expected sample size
        in the design and it is not a bootstrap sample.
    """
    pivoted_data = self._get_pivoted_and_trimmed_historical_data(
        start_week_id, bootstrap_sample=False
    )[self.design.primary_metric]

    if self.design.is_crossover:
      stacked_values = np.stack([
          pivoted_data["test_1"].values - pivoted_data["test_2"].values,
          pivoted_data["test_1"].values,
          pivoted_data["test_2"].values,
      ])
      # Trimming is performed based on the difference (first stacked array)
      # since that is how it's done in the analysis.
      trimmed_stacked_values = statistics.TrimmedArray(
          stacked_values, self.design.post_trim_percentile
      )
      trimmed_means = trimmed_stacked_values.mean()
      trimmed_var = trimmed_stacked_values.var(ddof=1)

      primary_metric_average = 0.5 * (trimmed_means[1] + trimmed_means[2])
      primary_metric_variance = trimmed_var[0]

      # If we have post trimming, then we don't demean the metrics (to ensure
      # that the control average remains sensible). This means we need to add
      # the extra variance introduced by the difference in the means in the two
      # periods.
      if self.design.post_trim_percentile > 0.0:
        primary_metric_variance += (
            trimmed_means[1] - trimmed_means[2]
        ) ** 2 / 4

      sample_size = len(trimmed_stacked_values)
    else:
      if "pretest" in pivoted_data.columns:
        pivoted_data["test"] = statistics.apply_cuped_adjustment(
            pivoted_data["test"].values,
            pivoted_data["pretest"].values,
            self.design.post_trim_percentile
        )

      primary_metric = statistics.TrimmedArray(
          pivoted_data["test"].values, self.design.post_trim_percentile
      )
      primary_metric_average = float(primary_metric.mean())
      primary_metric_variance = float(primary_metric.var(ddof=1))
      sample_size = len(primary_metric)

    if sample_size != self.design.n_items_after_post_trim:
      raise RuntimeError(
          "Unexpected number of items after post-trimming. Design expects "
          f"{self.design.n_items_after_post_trim:,}, but historical data has "
          f"{sample_size:,}"
      )

    return (
        primary_metric_average,
        primary_metric_variance,
        sample_size,
    )

  def _estimate_average_primary_metric_stats(self) -> tuple[float, float, int]:
    """Estimates the stats of the primary metric.

    This esimates the mean, standard deviation and sample size of the
    primary metric, by iterating over all valid start weeks and averaging
    the results.

    Returns:
      The average, standard deviation and sample size of the primary metric,
      averaged over all valid start dates in the historical data.

    Raises:
      RuntimeError: If the calculations do not have the same sample size for
        every start week. This may happen if the input data is missing rows.
    """
    primary_metric_stats = list(
        map(self._estimate_primary_metric_stats, self.valid_start_week_ids)
    )
    primary_metric_averages, primary_metric_variances, sample_sizes = zip(
        *primary_metric_stats
    )

    primary_metric_average = np.mean(primary_metric_averages)
    primary_metric_standard_deviation = np.sqrt(
        np.mean(primary_metric_variances)
    )
    sample_sizes = np.asarray(sample_sizes)
    sample_size = sample_sizes[0]

    if np.any(sample_sizes != sample_size):
      raise RuntimeError(
          "The sample size was different for different weeks. This shouldn't"
          f"happen, check the data isn't missing any rows. {sample_sizes = }."
      )

    return (
        primary_metric_average,
        primary_metric_standard_deviation,
        sample_size,
    )

  def _estimate_minimum_detectable_effect_from_stats(
      self,
      primary_metric_average: float,
      primary_metric_standard_deviation: float,
      sample_size: float,
  ) -> tuple[float, float]:
    """Estimates the minimum detectable effect from the experiment statistics.

    This returns both the minimum detectable effect (mde) and relative minimum
    detectable effect (relative mde), where the relative mde is the mde divided
    by the primary metric average.

    Args:
      primary_metric_average: The average of the primary metric.
      primary_metric_standard_deviation: The standard deviation of the primary
        metric.
      sample_size: The sample size of the primary metric.

    Returns:
      The estimated minimum detectable effect and relative minimum detectable
      effect.
    """

    if self.design.is_crossover:
      minimum_detectable_effect = (
          statistics.yuens_t_test_paired_minimum_detectable_effect(
              difference_standard_deviation=primary_metric_standard_deviation,
              sample_size=sample_size,
              alternative="two-sided",
              power=self.design.power,
              alpha=self.design.alpha,
          )
      )
    else:
      minimum_detectable_effect = (
          statistics.yuens_t_test_ind_minimum_detectable_effect(
              standard_deviation=primary_metric_standard_deviation,
              sample_size=sample_size,
              alternative="two-sided",
              power=self.design.power,
              alpha=self.design.alpha,
          )
      )

    relative_minimum_detectable_effect = (
        minimum_detectable_effect / primary_metric_average
    )

    return (
        float(minimum_detectable_effect),
        float(relative_minimum_detectable_effect),
    )

  def estimate_minimum_detectable_effect(self) -> None:
    """Estimates the minimum detectable effect for this experiment design.

    This uses the historical data to estimate the average, standard deviation
    and sample size of the primary metric for this design, and then based on
    that estimates the minimum detectable effect. To estimate the statistics
    for the primary metric, it averages over all of the valid start weeks for
    this experiment design and historical dataset.

    After running this method, the results will be stored in the following
    attributes:
      - primary_metric_average,
      - primary_metric_standard_deviation
      - minimum_detectable_effect
      - relative_minimum_detectable_effect
    """

    primary_metric_average, primary_metric_standard_deviation, sample_size = (
        self._estimate_average_primary_metric_stats()
    )

    minimum_detectable_effect, relative_minimum_detectable_effect = (
        self._estimate_minimum_detectable_effect_from_stats(
            primary_metric_average=primary_metric_average,
            primary_metric_standard_deviation=primary_metric_standard_deviation,
            sample_size=sample_size,
        )
    )

    self.primary_metric_average = primary_metric_average
    self.primary_metric_standard_deviation = primary_metric_standard_deviation
    self.minimum_detectable_effect = minimum_detectable_effect
    self.relative_minimum_detectable_effect = relative_minimum_detectable_effect

  def validate_design(
      self, n_simulations: int = 1000, progress_bar: bool = True
  ) -> None:
    """Runs a range of simulation tests to validate the experiment design.

    These simulations are designed to test that the historical data is meeting
    the distributional assumptions made by the statistical tests used. If these
    validations fail, then this design should not be used.

    This simulates many A/A experiments and synthetic A/B experiments to
    perform the validation.

    The A/A experiments are simulated by:
      1. Sampling the historical data with replacement.
      2. Selecting a start week id at random.
      3. Randomly splitting the items into control and treatment.
      4. Analyzing the primary metric based on the experiment design.

    The A/B experiments are simulated by:
      1. Sampling the historical data with replacement.
      2. Selecting a start week id at random.
      3. Randomly splitting the items into control and treatment.
      4. Applying a synthetic treatment effect to the primary metric when it
         is treated. The effect size is the estimated minimum detectable
         effect.
      5. Analyzing the primary metric based on the experiment design.

    The validation checks the following:

    The A/A tests should have uniformly distributed p-values, and should return
    a statistically significant result design.alpha percent of the time. The
    absolute difference between the primary metric in control and treatment
    should be 0 on average.

    The A/B tests should return a statistically significant result design.power
    percent of the time. The absolute difference between the primary metric in
    control and treatment should be equal to the estimated minimum detectable
    effect.

    Args:
      n_simulations: The number of experiments to simulate. Defaults to 1000.
      progress_bar: Whether to show a progress bar or not. Defaults to True.

    Raises:
      RuntimeError: If this is called before the
        estimate_minimum_detectable_effect() method.
    """
    if self.minimum_detectable_effect is None:
      raise RuntimeError(
          "Cannot run validate_design() before"
          " estimate_minimum_detectable_effect()."
      )

    aa_simulation_results_list = []
    ab_simulation_results_list = []

    if progress_bar:
      simulations_iterator = tqdm.trange(
          n_simulations, desc=f"Validating {self.design.design_id}"
      )
    else:
      simulations_iterator = range(n_simulations)

    for _ in simulations_iterator:
      start_week_id = self.rng.choice(self.valid_start_week_ids)
      simulated_aa_test_data = (
          self._get_pivoted_and_trimmed_historical_data(
              start_week_id, bootstrap_sample=True
          )
          .reset_index()
          .pipe(
              apply_random_treatment_assignment,
              rng=self.rng,
              item_id_column=self.item_id_column,
              treatment_column=self.treatment_column,
          )
          .set_index([self.item_id_column, self.treatment_column])
      )

      aa_simulation_results_list.append(
          experiment_analysis.analyze_single_metric(
              simulated_aa_test_data,
              design=self.design,
              metric_name=self.design.primary_metric,
              treatment_assignment_index_name=self.treatment_column,
          )
      )

      simulated_ab_test_data = apply_synthetic_treatment_effect(
          simulated_aa_test_data,
          metric_name=self.design.primary_metric,
          design=self.design,
          effect_size=self.minimum_detectable_effect,
          treatment_assignment_index_name=self.treatment_column,
      )

      ab_simulation_results_list.append(
          experiment_analysis.analyze_single_metric(
              simulated_ab_test_data,
              design=self.design,
              metric_name=self.design.primary_metric,
              treatment_assignment_index_name=self.treatment_column,
          )
      )

    self.aa_simulation_results = pd.DataFrame.from_records(
        map(dataclasses.asdict, aa_simulation_results_list)
    )
    self.ab_simulation_results = pd.DataFrame.from_records(
        map(dataclasses.asdict, ab_simulation_results_list)
    )

  @property
  def null_p_value_robustness_p_value(self) -> float | None:
    """Returns a p-value testing that the A/A p-values are reliable.

    Reliable p-values must follow a uniform distribution under the null
    hypothesis (A/A tests). This uses a Kolmogorov–Smirnov test to test it.

    If the validate_design() method has not yet been run, this returns None.
    """
    if self.aa_simulation_results is None:
      return None

    return stats.kstest(
        self.aa_simulation_results["p_value"].values, stats.uniform.cdf
    ).pvalue

  @property
  def _aa_is_statistically_significant(self) -> np.ndarray | None:
    if self.aa_simulation_results is None:
      return None

    return (
        self.aa_simulation_results["p_value"].values < self.design.alpha
    ).astype(int)

  @property
  def simulated_false_positive_rate(self) -> float | None:
    """Returns the simulated false positive rate from the A/A tests.

    In A/A tests there is no true effect, so any statistically significant
    results are false positives. The false positive rate should be close to
    design.alpha.

    If the validate_design() method has not yet been run, this returns None.
    """
    if self.aa_simulation_results is None:
      return None

    return np.mean(self._aa_is_statistically_significant)

  @property
  def false_positive_rate_robustness_p_value(self) -> float | None:
    """Returns a p-value testing that the false positive rate is valid.

    We want to ensure that the false positive rate is not higher than the
    specified design.alpha. This is done by using a binomial statistical test.

    If the validate_design() method has not yet been run, this returns None.
    """
    if self.aa_simulation_results is None:
      return None

    return stats.binomtest(
        k=np.sum(self._aa_is_statistically_significant),
        n=len(self._aa_is_statistically_significant),
        p=self.design.alpha,
        alternative="greater",
    ).pvalue

  @property
  def _ab_is_statistically_significant(self) -> np.ndarray | None:
    if self.aa_simulation_results is None:
      return None

    return (
        self.ab_simulation_results["p_value"].values < self.design.alpha
    ).astype(int)

  @property
  def simulated_power_at_minimum_detectable_effect(self) -> float | None:
    """Returns the power where the true effect is the minimum detectable effect.

    In A/B tests we simulated a true effect of the size of the minimum
    detectable effect, so a significant result is a true positive. The fraction
    of the simulations with a statistically significant result is the true
    positive rate, also known as the power. This should be close to
    design.power.

    If the validate_design() method has not yet been run, this returns None.
    """
    if self.aa_simulation_results is None:
      return None

    return np.mean(self._ab_is_statistically_significant)

  @property
  def power_robustness_p_value(self) -> float | None:
    """Returns a p-value testing that the simulated power is valid.

    In A/B tests we simulated a true effect of the size of the minimum
    detectable effect. We want to ensure that at the minimum detectable effect
    we can measure a statistically significant result at least design.power
    percent of the time.

    If the validate_design() method has not yet been run, this returns None.
    """
    if self.aa_simulation_results is None:
      return None

    return stats.binomtest(
        k=np.sum(self._ab_is_statistically_significant),
        n=len(self._ab_is_statistically_significant),
        p=self.design.power,
        alternative="less",
    ).pvalue

  @property
  def aa_point_estimate_robustness_p_value(self) -> float | None:
    """Returns a p-value testing that the A/A test point estimates are valid.

    This is valid if the distribution of point estimates from the A/A tests
    is centered on 0. This is tested with a 1 sample t-test.

    If the validate_design() method has not yet been run, this returns None.
    """
    if self.aa_simulation_results is None:
      return None

    return stats.ttest_1samp(
        self.aa_simulation_results["absolute_difference"].values,
        0.0,
    ).pvalue

  @property
  def ab_point_estimate_robustness_p_value(self) -> float | None:
    """Returns a p-value testing that the A/B test point estimates are valid.

    This is valid if the distribution of point estimates from the A/B tests
    is centered on the minimum detectable effect. This is tested with a 1 sample
    t-test.

    If the validate_design() method has not yet been run, this returns None.
    """
    if self.aa_simulation_results is None:
      return None

    return stats.ttest_1samp(
        self.ab_simulation_results["absolute_difference"].values,
        self.minimum_detectable_effect,
    ).pvalue

  def _is_above_p_value_threshold(self, p_value: float | None) -> bool | None:
    """Compares a p-value with the robustness_p_value_threshold attribute.

    Returns None if the p-value is None, True if the p-value is greater or
    equal to the threshold and false otherwise.
    """

    if p_value is None:
      return None

    return p_value >= self.robustness_p_value_threshold

  @property
  def null_p_value_robustness_check_pass(self) -> bool | None:
    """Does this design pass the null p-values robustness check."""
    return self._is_above_p_value_threshold(
        self.null_p_value_robustness_p_value
    )

  @property
  def power_robustness_check_pass(self) -> bool | None:
    """Does this design pass the power robustness check."""
    return self._is_above_p_value_threshold(self.power_robustness_p_value)

  @property
  def aa_point_estimate_robustness_check_pass(self) -> bool | None:
    """Does this design pass the A/A point estimate robustness check."""
    return self._is_above_p_value_threshold(
        self.aa_point_estimate_robustness_p_value
    )

  @property
  def ab_point_estimate_robustness_check_pass(self) -> bool | None:
    """Does this design pass the A/B point estimate robustness check."""
    return self._is_above_p_value_threshold(
        self.ab_point_estimate_robustness_p_value
    )

  @property
  def false_positive_rate_robustness_check_pass(self) -> bool | None:
    """Does this design pass the false positive rate robustness check."""
    return self._is_above_p_value_threshold(
        self.false_positive_rate_robustness_p_value
    )

  @property
  def all_robustness_checks_pass(self) -> bool | None:
    """Does this design pass all robustness checks."""
    robustness_checks = [
        self.null_p_value_robustness_check_pass,
        self.power_robustness_check_pass,
        self.false_positive_rate_robustness_check_pass,
        self.aa_point_estimate_robustness_check_pass,
        self.ab_point_estimate_robustness_check_pass,
    ]
    if any([check is None for check in robustness_checks]):
      return None
    else:
      return all(robustness_checks)

  def summary_dict(self) -> dict[str, str | int | float | bool | None]:
    """Returns the main results as a dictionary.

    The dictionary will contain:
      - all of the paramters of the experiment design
      - minimum_detectable_effect
      - relative_minimum_detectable_effect
      - simulated_false_positive_rate
      - simulated_power_at_minimum_detectable_effect
      - null_p_value_robustness_check_pass (True/False)
      - power_robustness_check_pass (True/False)
      - false_positive_rate_robustness_check_pass (True/False)
      - aa_point_estimate_robustness_check_pass (True/False)
      - ab_point_estimate_robustness_check_pass (True/False)
      - all_robustness_checks_pass (True/False)

    If estimate_minimum_detectable_effect() has not been run, then all of the
    values that are not from the experiment design will be None.

    If estimate_minimum_detectable_effect() has been run but
    validate_experiment() has not, then everything except the design parameters,
    minimum_detectable_effect and relative_minimum_detectable_effect will be
    None.
    """

    summary_dict = dataclasses.asdict(self.design)
    summary_dict["design_id"] = self.design.design_id
    del(summary_dict["coinflip_salt"])  # Not relevant for the summary

    summary_metrics = [
        "minimum_detectable_effect",
        "relative_minimum_detectable_effect",
        "simulated_false_positive_rate",
        "simulated_power_at_minimum_detectable_effect",
        "null_p_value_robustness_check_pass",
        "power_robustness_check_pass",
        "false_positive_rate_robustness_check_pass",
        "aa_point_estimate_robustness_check_pass",
        "ab_point_estimate_robustness_check_pass",
        "all_robustness_checks_pass",
    ]

    for metric in summary_metrics:
      summary_dict[metric] = getattr(self, metric)

    return summary_dict


def make_analysis_summary_dataframe(
    experiment_analyses: Collection[SimulationAnalysis],
    sort_by: str = "relative_minimum_detectable_effect",
) -> pd.DataFrame:
  """Returns a dataframe summarising a list of simulation analysis results.

  Args:
    experiment_analyses: The list of experiment analyses.
    sort_by: The column to sort the resulting dataframe by. Defaults to the
      relative_minimum_detectable_effect.
  """
  return (
      pd.DataFrame.from_records(
          list(map(lambda x: x.summary_dict(), experiment_analyses))
      )
      .set_index("design_id")
      .sort_values(sort_by)
  )


def _format_check_column(value: bool | None) -> str:
  if value is None:
    return ""
  return (
      "background-color:lightgreen"
      if value
      else "background-color:darkred;color:white"
  )


def _drop_validation_columns(summary_data: pd.DataFrame) -> pd.DataFrame:
  validation_columns = [
      column
      for column in summary_data.columns
      if column.endswith("_pass") | column.startswith("simulated_")
  ]
  return summary_data.drop(columns=validation_columns)


def _extract_constant_design_columns_for_caption(
    summary_data: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
  """Removes any constant design columns and adds them to a caption.

  Any design columns that are the same for every row in the summary data are
  dropped from the dataframe, and added to a string that can be used as the
  dataframe caption.

  Args:
    summary_data: The data to extract constant columns from.

  Returns:
    The data with the constant columns dropped, and a string containing a
    summary of the constant columns.
  """
  design_attributes = [
      "primary_metric",
      "alpha",
      "power",
      "n_items_before_trimming",
      "n_items_after_pre_trim",
      "n_items_after_post_trim",
      "runtime_weeks",
      "pretest_weeks",
      "pre_trim_top_percentile",
      "pre_trim_bottom_percentile",
      "post_trim_percentile",
      "is_crossover",
  ]
  constant_design_attributes = []
  constant_attributes_summaries = []
  for column in design_attributes:
    if summary_data[column].nunique(dropna=False) == 1:
      constant_design_attributes.append(column)
      constant_attributes_summaries.append(
          f"{column} = {summary_data.iloc[0][column]}"
      )

  summary_data = summary_data.drop(columns=constant_design_attributes)
  constant_attributes_summary = ", ".join(constant_attributes_summaries)

  return summary_data, constant_attributes_summary


def _clean_column_names(summary_data: pd.DataFrame) -> pd.DataFrame:
  summary_data.columns = [
      column.replace("_", " ").title() for column in summary_data.columns
  ]
  return summary_data


def format_analysis_summary_dataframe(
    summary_data: pd.DataFrame, with_validation_columns: bool = True
) -> style.Styler:
  """Formats the summary dataframe for nice displaying in jupyter notebooks.

  Args:
    summary_data: The data to be displayed, should have been generated with
      make_analysis_summary_dataframe().
    with_validation_columns: Whether to display the validation columns. Defaults
      to True.

  Returns:
    The styled dataframe.
  """
  summary_data = summary_data.copy()

  if not with_validation_columns:
    summary_data = _drop_validation_columns(summary_data)

  summary_data, constant_attributes_summary = (
      _extract_constant_design_columns_for_caption(summary_data)
  )
  caption = (
      "Analysis results of experiment designs where the following design"
      f" attributes are constant for all designs: {constant_attributes_summary}"
  )

  summary_data = _clean_column_names(summary_data)

  def _get_existing_summary_columns(columns: Collection[str]) -> list[str]:
    return list(filter(lambda x: x in summary_data.columns.values, columns))

  styled_summary_data = (
      summary_data.style.set_caption(caption)
      .bar(
          subset=_get_existing_summary_columns([
              "Minimum Detectable Effect",
              "Relative Minimum Detectable Effect",
          ]),
          cmap=mpl.colormaps["RdYlGn_r"],
      )
      .format(
          "{:.2%}",
          subset=_get_existing_summary_columns([
              "Relative Minimum Detectable Effect",
              "Pre Trim Top Percentile",
              "Pre Trim Bottom Percentile",
              "Post Trim Percentile",
              "Simulated False Positive Rate",
              "Simulated Power At Minimum Detectable Effect",
          ]),
      )
      .highlight_null(props="opacity:0%")
      .applymap(
          _format_check_column,
          subset=[
              column
              for column in summary_data.columns
              if column.endswith(" Pass")
          ],
      )
  )

  return styled_summary_data
