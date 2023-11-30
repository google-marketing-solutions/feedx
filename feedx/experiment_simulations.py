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

import numpy as np
import pandas as pd
import tqdm.autonotebook as tqdm

from feedx import experiment_analysis
from feedx import experiment_design
from feedx import statistics

ExperimentDesign = experiment_design.ExperimentDesign


def generate_all_valid_designs(
    max_items_for_test: Collection[int],
    crossover_design_allowed: bool,
    traditional_design_allowed: bool,
    candidate_runtime_weeks: Collection[int],
    candidate_pretest_weeks: Collection[int],
    candidate_pre_trim_percentiles: Collection[float],
    candidate_post_trim_percentiles: Collection[float],
    primary_metric: str,
    crossover_washout_weeks: int | None = None,
) -> list[ExperimentDesign]:
  """Generate valid experiment designs with user-defined parameters.

  Args:
    max_items_for_test: number of items to test
    crossover_design_allowed: crossover experiment design is allowed
    traditional_design_allowed: traditional experiment design is allowed
    candidate_runtime_weeks: number of weeks to run the experiment
    candidate_pretest_weeks: number of weeks for data prior to the experiment
    candidate_pre_trim_percentiles: percetage at which to omit the lowest and 
      highest values of the primary metric from the experiment, eg remove 1% of 
      the lowest and 1% of the highest values from the experiment
    candidate_post_trim_percentiles: percetage at which to omit the lowest and 
      highest values of the primary metric from the analysis, eg remove 1% of 
      the lowest and 1% of the highest values from the analysis
    primary_metric: ads performance metric on which to base experiment design
    crossover_washout_weeks: number of weeks to exclude at the start of each
      crossover period

  Returns:
    List of valid experiment designs
  """
  if crossover_design_allowed & (crossover_washout_weeks is None):
    raise ValueError(
        "When allowing a crossover design you must specify the number of"
        " crossover washout weeks."
    )

  valid_designs = []
  design_constants = dict(
      primary_metric=primary_metric,
  )
  design_inputs = itertools.product(
      max_items_for_test,
      candidate_runtime_weeks,
      candidate_pretest_weeks,
      candidate_pre_trim_percentiles,
      candidate_post_trim_percentiles,
  )

  for (
      n_items,
      runtime_weeks,
      pretest_weeks,
      pre_trim_percentile,
      post_trim_percentile,
  ) in design_inputs:
    if (pre_trim_percentile > 0) & (pretest_weeks == 0):
      # Cannot do pretrimming without pretest weeks.
      continue

    if traditional_design_allowed:
      valid_designs.append(
          ExperimentDesign(
              n_items_before_trimming=n_items,
              runtime_weeks=runtime_weeks,
              pretest_weeks=pretest_weeks,
              pre_trim_top_percentile=pre_trim_percentile,
              pre_trim_bottom_percentile=0.0,
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
              n_items_before_trimming=n_items,
              runtime_weeks=runtime_weeks,
              pretest_weeks=pretest_weeks,
              pre_trim_top_percentile=pre_trim_percentile,
              pre_trim_bottom_percentile=0.0,
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
    sample_weight_column: str = "n_samples_",
    dummy_list_column: str = "dummy_list_",
) -> pd.DataFrame:
  """Bootstrap sample the data.

  This creates a bootstrap sampled copy of the data, which is the equivalent
  of sampling the same number of rows as in the original set, with replacement.

  Instead of doing the actual sampling, we use poisson bootstrapping, which
  is more efficient but approximate (it won't always return exactly the same
  sample size as the original data). However for large samples it's a good
  approximation and it's more efficient.

  Args:
    data: The dataset to sample.
    rng: A random number generator to perform the sampling.
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
        f"{protected_columns}"
    )

  copied_data = data.copy()
  copied_data[sample_weight_column] = rng.poisson(
      lam=1.0, size=len(copied_data.index.values)
  )
  copied_data[dummy_list_column] = copied_data[sample_weight_column].apply(
      np.arange
  )
  exploded_data = (
      copied_data.explode(dummy_list_column)
      .dropna(axis=0, subset=[dummy_list_column])
      .drop(columns=[dummy_list_column, sample_weight_column])
  )
  return exploded_data


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
    pivoted_data: pd.DataFrame, effect_size: float, treatment_column: str
) -> pd.DataFrame:
  is_treated = pivoted_data[treatment_column] == 1
  pivoted_data.loc[is_treated, "test"] += effect_size
  return pivoted_data


def _apply_synthetic_treatment_effect_to_crossover_experiment(
    pivoted_data: pd.DataFrame, effect_size: float, treatment_column: str
) -> pd.DataFrame:
  is_treated = pivoted_data[treatment_column] == 1
  pivoted_data.loc[is_treated, "test_1"] += effect_size
  pivoted_data.loc[~is_treated, "test_2"] += effect_size
  return pivoted_data


def apply_synthetic_treatment_effect(
    pivoted_data: pd.DataFrame,
    *,
    design: ExperimentDesign,
    effect_size: float,
    treatment_column: str,
) -> pd.DataFrame:
  """Applies the synthetic treatment effect to the dataframe.

  If the design is a crossover design, then the pivoted_data should contain
  columns named "test_1" and "test_2" which contain the metric to add a
  synthetic treatment effect to. If the design is a regular design then the
  pivoted_data should contain a single column named "test" which contains the
  metric to apply the synthetic treatment effect to.

  The synthetic treatment will increase the metric by the effect_size for
  treatment when they have treatment applied. For a regular experiment, this
  is all the items where treatment_column = 1. For a crossover experiment, this
  is all the items where treatment_column = 1 for test_1, and all the columns
  where treatment_column = 0 for test_2.

  Args:
    pivoted_data: The dataframe to apply the synthetic treatment to. Must
      include the treatment_column, as well as "test" if it is not a crossover
      design, or "test_1" and "test_2" if it is a crossover design.
    design: The experiment design being simulated, with the is_crossover
      attribute dictating whether this is a crossover design or regular design.
    effect_size: The effect size to apply to the metric when it is treated.
    treatment_column: The column name that has the treatment status in the data.

  Returns:
    The dataframe with the synthetic treatment effect applied.

  Raises:
    ValueError: If the pivoted_data is missing any required columns.
  """
  required_columns = {treatment_column}
  if design.is_crossover:
    required_columns.update({"test_1", "test_2"})
  else:
    required_columns.add("test")

  missing_columns = required_columns - set(pivoted_data.columns)
  if missing_columns:
    raise ValueError(
        "The pivoted_data is missing the following required columns: "
        f"{missing_columns}"
    )

  if design.is_crossover:
    return _apply_synthetic_treatment_effect_to_crossover_experiment(
        pivoted_data, effect_size, treatment_column
    )
  else:
    return _apply_synthetic_treatment_effect_to_regular_experiment(
        pivoted_data, effect_size, treatment_column
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
    aa_robustness_pvalue: A p-value testing the null hypothesis that the A/A
      simulation results are valid. None if the validation has not been run.
    ab_robustness_pvalue: A p-value testing the null hypothesis that the A/B
      simulation results are valid. None if the validation has not been run.
    power_at_minimum_detectable_effect: The actual simulated power for the
      minimum detectable effect, should be close to design.power. None if the
      validation has not been run.
    false_positive_rate: The actual simulated false positive rate, should be
      close to design.alpha. None if the validation has not been run.
    aa_simulation_results: The full A/A simulation results. None if the
      validation has not been run.
    ab_simulation_results: The full A/B simulation results. None if the
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

  aa_robustness_pvalue: float | None = dataclasses.field(
      default=None, init=False
  )
  ab_robustness_pvalue: float | None = dataclasses.field(
      default=None, init=False
  )
  power_at_minimum_detectable_effect: float | None = dataclasses.field(
      default=None, init=False
  )
  false_positive_rate: float | None = dataclasses.field(
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
    )[self.design.primary_metric]

    if bootstrap_sample:
      pivoted_data = _bootstrap_sample_data(pivoted_data, self.rng)

    if (not bootstrap_sample) and (
        len(pivoted_data.index.values) != self.design.n_items_before_trimming
    ):
      raise RuntimeError(
          "Unexpected number of items before trimming. Design expects "
          f"{self.design.n_items_before_trimming:,}, but historical data has "
          f"{len(pivoted_data.index.values):,}"
      )

    if self.design.pretest_weeks > 0:
      pivoted_data = experiment_analysis.trim_outliers(
          pivoted_data,
          order_by="pretest",
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
    )

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
      sample_size = len(trimmed_stacked_values)
    else:
      if "pretest" in pivoted_data.columns:
        pivoted_data["test"] = statistics.apply_cuped_adjustment(
            pivoted_data["test"].values, pivoted_data["pretest"].values
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
      )

      aa_simulation_results_list.append(
          experiment_analysis.analyze_experiment(
              simulated_aa_test_data, self.design
          )
      )

      simulated_ab_test_data = apply_synthetic_treatment_effect(
          simulated_aa_test_data,
          design=self.design,
          effect_size=self.minimum_detectable_effect,
          treatment_column=self.treatment_column,
      )

      ab_simulation_results_list.append(
          experiment_analysis.analyze_experiment(
              simulated_ab_test_data, self.design
          )
      )

    self.aa_simulation_results = pd.DataFrame.from_records(
        map(dataclasses.asdict, aa_simulation_results_list)
    )
    self.ab_simulation_results = pd.DataFrame.from_records(
        map(dataclasses.asdict, ab_simulation_results_list)
    )
