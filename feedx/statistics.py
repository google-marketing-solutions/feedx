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

"""A collection of functions for statistical inference of feedx experiments."""

import dataclasses
import functools
import numpy as np
import numpy.typing as npt
from scipy import stats


class TrimmedArray:
  """An array with support for calculating trimmed statistics.

  This is used for implementing the trimmed t-tests, also known as Yuen's
  t-test, which essentially follows these steps:

    1. Replace the normal means with the trimmed means
    2. Replace the normal standard deviation with the winsorized standard
      deviation (where the sample size is the number of samples after trimming)
    3. Replace the number of samples with the number of samples after trimming
    4. Apply the t-test

  References:
    https://www.jstor.org/stable/2334550
    https://www.jstor.org/stable/40063824
    https://real-statistics.com/students-t-distribution/problems-data-t-tests/yuen-welchs-test/

  Attributes:
    quantile: The quantile to trim from each side.
    original_sample_size: The untrimmed sample size of the array.
    n_trim: The number of samples to trim from each side.
    values_trimmed: The trimmed array.
    min_values: The minimum value of the trimmed array.
    max_values: The maximum value of the trimmed array.
  """

  def __init__(self, values: np.ndarray, quantile: float):
    """Initialises the trimmed array.

    Args:
      values: Input array to be trimmed, can be 1 or 2 dimensional. If it's 2
        dimensional then columns (axis=1) will be trimmed based on the extremes
        of the first row (axis=0), and all summary statistics will be calculated
        along the columns (axis=1).
      quantile: The quantile to trim from each side (between 0 and 0.5).

    Raises:
      ValueError: If quantile is not between 0 and 0.5.
    """

    if (quantile < 0.0) | (quantile >= 0.5):
      raise ValueError("Must use 0 <= quantile < 0.5")

    if len(values.shape) not in [1, 2]:
      raise ValueError(f"Must use 1 or 2 dimensional arrays, {values.shape = }")

    self._values_untrimmed = np.atleast_2d(values)
    self.quantile = quantile

    assert len(self) == self.original_sample_size - 2 * self.n_trim

  @functools.cached_property
  def _is_2d(self) -> bool:
    return self._values_untrimmed.shape[0] > 1

  @functools.cached_property
  def original_sample_size(self) -> int:
    return self._values_untrimmed.shape[-1]

  @functools.cached_property
  def n_trim(self) -> int:
    return int(np.floor(self.quantile * self.original_sample_size))

  @functools.cached_property
  def values_trimmed(self) -> np.ndarray:
    if self.n_trim == 0:
      trimmed_index = np.arange(self.original_sample_size)
    else:
      trimmed_index = np.sort(
          np.argsort(self._values_untrimmed[0, :])[self.n_trim : -self.n_trim]
      )

    return self._values_untrimmed[:, trimmed_index]

  @functools.cached_property
  def min_values(self) -> np.ndarray:
    return self.values_trimmed[:, np.argmin(self.values_trimmed[0, :])]

  @functools.cached_property
  def max_values(self) -> np.ndarray:
    return self.values_trimmed[:, np.argmax(self.values_trimmed[0, :])]

  def __str__(self) -> str:
    return str(self.values_trimmed)

  def __repr__(self) -> str:
    return repr(self.values_trimmed)

  def __len__(self) -> int:
    return len(self.values_trimmed[0, :])

  def mean(self) -> np.ndarray | float:
    """Calculates the mean of the trimmed array.

    The trimmed mean is the mean of all the values excluding the trimmed values.

    Example:
      Original array = [1.0, 4.0, 5.0, 7.0, 9.0, 12.0, 20.0]
      Trimmed array (trim 2 samples from each side):
        [5.0, 7.0, 9.0]
      Trimmed mean = mean([5.0, 7.0, 9.0])

    Returns:
      The trimmed mean of the array.
    """
    mean = np.mean(self.values_trimmed, axis=1)

    if self._is_2d:
      return mean
    else:
      return float(mean[0])

  def winsorized_mean(self) -> np.ndarray | float:
    """Calculates the winsorized mean.

    The winsorized mean is the mean of all the values, where the trimmed values
    are kept but they are clipped to the most extreme non-trimmed value.

    Example:
      Original array = [1.0, 4.0, 5.0, 7.0, 9.0, 12.0, 20.0]
      Winsorized array (trim 2 samples from each side):
        [5.0, 5.0, 5.0, 7.0, 9.0, 9.0, 9.0]
      Windorized mean = mean([5.0, 5.0, 5.0, 7.0, 9.0, 9.0, 9.0])

    Returns:
      The winsorized mean of the array.
    """
    winsorized_mean = (
        np.sum(self.values_trimmed, axis=1)
        + self.n_trim * (self.min_values + self.max_values)
    ) / self.original_sample_size

    if self._is_2d:
      return winsorized_mean
    else:
      return float(winsorized_mean[0])

  def cov(self, ddof: int = 0) -> np.ndarray | float:
    """Calculates the winsorized covariance matrix.

    The winsorized covariance between two arrays is calculated as:

    cov(x,y) =
      sum((y_trimmed - winsorized_mean(y)) * (x_trimmed - winsorized_mean(x)))
      / n_samples_after_trimming - ddof

    The diagonal of this matrix is the winsorized variance.

    Args:
      ddof: Means “Delta Degrees of Freedom”. The divisor used in the
        calculation is N - ddof, where N represents the number of elements. By
        default ddof is zero.

    Returns:
      The winsorized covariance matrix of the array for a 2d matrix, or the
        variance for a 1d array.
    """
    resid = self.values_trimmed - np.reshape(self.winsorized_mean(), (-1, 1))
    min_resid = (self.min_values - self.winsorized_mean()).reshape(-1, 1)
    max_resid = (self.max_values - self.winsorized_mean()).reshape(-1, 1)

    sum_resid_sq = resid @ resid.T + self.n_trim * (
        min_resid @ min_resid.T + max_resid @ max_resid.T
    )

    cov = sum_resid_sq / (len(self) - ddof)

    if self._is_2d:
      return cov
    else:
      return float(cov.flatten()[0])

  def var(self, ddof: int = 0) -> np.ndarray | float:
    """Calculates the winsorized variance.

    The winsorized variance is the sum of the squared winsorized residuals
    divided by the number of samples left after trimming - ddof.

    Example:
      Original array = [1.0, 4.0, 5.0, 7.0, 9.0, 12.0, 20.0]
      Winsorized array (trim 2 samples from each side):
        [5.0, 5.0, 5.0, 7.0, 9.0, 9.0, 9.0]
      Winsorized mean = mean([5.0, 5.0, 5.0, 7.0, 9.0, 9.0, 9.0]) = 7.0
      Winsorized residuals = [-2.0, -2.0, -2.0, 0.0, 2.0, 2.0, 2.0]
      Squared winsorized residuals = [4.0, 4.0, 4.0, 0.0, 4.0, 4.0, 4.0]
      Variance = sum([4.0, 4.0, 4.0, 0.0, 4.0, 4.0, 4.0]) / (3 - 1)

    Args:
      ddof: Means “Delta Degrees of Freedom”. The divisor used in the
        calculation is N - ddof, where N represents the number of elements. By
        default ddof is zero.

    Returns:
      The winsorized variance of the array.
    """
    if self._is_2d:
      return np.diag(self.cov(ddof=ddof))
    else:
      return self.cov(ddof=ddof)

  def std(self, ddof: int = 0) -> np.ndarray | float:
    """Calculates the winsorized standard deviation.

    This is calculated as the square root of the winsorized variance.

    Args:
      ddof: Means “Delta Degrees of Freedom”. The divisor used in the
        calculation is N - ddof, where N represents the number of elements. By
        default ddof is zero.

    Returns:
      The winsorized standard deviation of the array.
    """
    return np.sqrt(self.var(ddof=ddof))


def apply_cuped_adjustment(
    metric: np.ndarray, covariate: np.ndarray, trimming_quantile: float
) -> np.ndarray:
  """Applies CUPED to control for pre-experiment covariates.

  This method applies a CUPED ajustment to a 1-d array. This adjustment reduces
  the (trimmed) variance by adjusting the metric with the covariate, while
  keeping the (trimmed) mean unchanged.

  Note: trimming is only applied when calculating the means and covariance
  matrix, it is not applied to the final output, meaning the output
  still contains the complete sample of data and must be trimmed again if
  required.

  The formula for CUPED is:

    cuped_metric = (
        metric -
        cov(metric, covariate) / var(covariate) * (covariate - mean(covariate))
    )

  After the CUPED adjustment, the (trimmed) mean of the metric will be
  unchanged, but the variance should be reduced. The amount by which the
  variance is reduced depends on the correlation between the covariate and
  the metric:

    var(cuped_metric) = var(metric) * (1 - corr(metric, covariate)**2)

  References:
    https://ai.stanford.edu/users/ronnyk.link/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf

  Args:
    metric: An array containing the metric to adjust.
    covariate: An array containing the covariate used to perform the adjustment.
    trimming_quantile: The fraction of samples to trim from each side of the
      metric when calculating the means and covariance matrix.

  Returns:
    The cuped adjusted metric.
  """
  metric = np.asarray(metric)
  covariate = np.asarray(covariate)

  metric_and_covariate = np.stack([metric, covariate], axis=0)
  covariance_matrix = np.cov(metric_and_covariate)
  theta = (
      covariance_matrix[0, 1] / covariance_matrix[1, 1]
  )
  cuped_metric = metric - theta * (covariate - np.mean(covariate))

  trimmed_metric_and_covariate = TrimmedArray(
      np.stack([cuped_metric, metric, covariate], axis=0),
      trimming_quantile
  )
  if trimmed_metric_and_covariate.n_trim == 0:
    return cuped_metric

  trimmed_covariance_matrix = trimmed_metric_and_covariate.cov()
  trimmed_theta = (
      trimmed_covariance_matrix[1, 2] / trimmed_covariance_matrix[2, 2]
  )
  cuped_metric_no_offset = metric - trimmed_theta * covariate

  trimmed_metric = TrimmedArray(metric, trimming_quantile)
  trimmed_cuped_metric_no_offset = TrimmedArray(
      cuped_metric_no_offset, 
      trimming_quantile
  )
  cuped_metric = (
      cuped_metric_no_offset
      + trimmed_metric.mean()
      - trimmed_cuped_metric_no_offset.mean()
  )

  if trimmed_cuped_metric_no_offset.var() < trimmed_metric.var():
    return cuped_metric
  else:
    return metric


@dataclasses.dataclass
class StatisticalTestResults:
  """Contains the results of a statistical test.

  Attributes:
    is_significant: Is the test statistically significant at the given alpha
      level.
    alpha: The alpha level used for the test. The result is statistically
      significant if the p_value < alpha.
    p_value: The p-value of the statistical test.
    statistic: The test statistic used to calculate the p-value.
    absolute_difference: The point estimated of the absolute difference between
      the two samples.
    absolute_difference_lower_bound: The lower bound of the absolute difference
      between the two samples.
    absolute_difference_upper_bound: The upper bound of the absolute difference
      between the two samples.
    relative_difference: The point estimated of the relative difference between
      the two samples. If the metric is not always >=0, then this is not
      calculated and set to None.
    relative_difference_lower_bound: The lower bound of the relative difference
      between the two samples. If the metric is not always >=0, then this is not
      calculated and set to None.
    relative_difference_upper_bound: The upper bound of the relative difference
      between the two samples. If the metric is not always >=0, then this is not
      calculated and set to None.
    standard_error: The standard error of the absolute difference between the
      two samples.
    sample_size: The size of the sample of data from which these statistics were
      calculated.
    degrees_of_freedom: The degrees of freedom of the statistical test.
  """

  is_significant: bool
  alpha: float
  p_value: float
  statistic: float
  absolute_difference: float
  absolute_difference_lower_bound: float
  absolute_difference_upper_bound: float
  relative_difference: float | None
  relative_difference_lower_bound: float | None
  relative_difference_upper_bound: float | None
  standard_error: float
  sample_size: int
  degrees_of_freedom: int | float
  control_average: float


def _ttest_from_stats(
    point_estimate: float,
    standard_error: float,
    sample_size: int,
    degrees_of_freedom: int | float,
    alternative: str,
) -> tuple[float, float]:
  """A helper function to perform the t-test.

  Performs inference on the statistic:

    t = point_estimate / standard_error

  Where t is assumed to follow a t-distribution with df=degrees_of_freedom.

  Args:
    point_estimate: The point estimate being evaluated.
    standard_error: The standard error of the point estimate.
    sample_size: The size of the sample used to estimate the point estimate.
    degrees_of_freedom: The estimated degrees of freedom of the estimate.
    alternative: The alternative hypothesis to test, one of ['two-sided',
      'greater', 'less']

  Returns:
    The test statistic and the p value.

  Raises:
    ValueError: If the alternative is not one of the expected values.
  """
  statistic = point_estimate / standard_error

  if alternative == "two-sided":
    p_value = 2.0 * stats.t.cdf(-1 * np.abs(statistic), df=degrees_of_freedom)
  elif alternative == "greater":
    p_value = stats.t.cdf(-1 * statistic, df=degrees_of_freedom)
  elif alternative == "less":
    p_value = stats.t.cdf(statistic, df=degrees_of_freedom)
  else:
    raise ValueError(
        "Alternative must be one of ['two-sided', 'greater', 'less']"
    )

  return statistic, p_value


def _absolute_difference_confidence_interval(
    *,
    mean_1: float,
    mean_2: float,
    standard_error_1: float,
    standard_error_2: float,
    corr: float,
    degrees_of_freedom: int | float,
    alternative: str,
    alpha: float = 0.05,
) -> tuple[float, float]:
  """Estimates the confidence interval for the absolute difference.

  This estimates the confidence interval for the absolute difference, defined
  as:

    absolute_difference = mean(sample_1) - mean(sample_2)

  The confidence interval is constructed to match the alternative hypothesis
  being tested. So if the alternative is "less", then the lower bound will be
  -infinity, while if the alternative is "greater" then the upper bound will be
  infinity. If the alternative is "two-sided" then a regular confidence
  interval is constructed.

  Args:
    mean_1: The mean of the first sample.
    mean_2: The mean of the second sample.
    standard_error_1: The standard error on mean_1.
    standard_error_2: The standard error on mean_2.
    corr: The correlation between the first and second sample.
    degrees_of_freedom: The degrees of freedom of the estimates.
    alternative: The alternative hypothesis to test, one of ['two-sided',
      'greater', 'less']
    alpha: The alpha level of the confidence interval, defaults to 0.05.

  Returns:
    A tuple containing the lower and upper bounds, (lower, upper).

  Raises:
    ValueError: If the alternative is not one of ['two-sided', 'greater',
      'less'].
  """
  var_1 = standard_error_1**2
  var_2 = standard_error_2**2
  var_12 = standard_error_1 * standard_error_2 * corr

  absolute_diff = mean_1 - mean_2
  absolute_diff_standard_error = np.sqrt(var_1 + var_2 - 2 * var_12)

  if alternative == "two-sided":
    critical_t_value = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha / 2.0)
    lower_bound = (
        absolute_diff - critical_t_value * absolute_diff_standard_error
    )
    upper_bound = (
        absolute_diff + critical_t_value * absolute_diff_standard_error
    )
  elif alternative == "greater":
    critical_t_value = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha)
    lower_bound = (
        absolute_diff - critical_t_value * absolute_diff_standard_error
    )
    upper_bound = np.inf
  elif alternative == "less":
    critical_t_value = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha)
    lower_bound = -np.inf
    upper_bound = (
        absolute_diff + critical_t_value * absolute_diff_standard_error
    )
  else:
    raise ValueError(
        "Alternative must be one of ['two-sided', 'greater', 'less']"
    )

  return lower_bound, upper_bound


def _relative_difference_confidence_interval(
    *,
    mean_1: float,
    mean_2: float,
    standard_error_1: float,
    standard_error_2: float,
    corr: float,
    degrees_of_freedom: int | float,
    alternative: str,
    alpha: float = 0.05,
) -> tuple[float, float]:
  """Estimates the confidence interval for the relative difference.

  This uses Fieller's theorem to estimate the confidence interval for the
  relative difference, where the relative difference defined as:

    relative_difference = (mean(sample_1) - mean(sample_2)) / mean(sample_2)
                        = mean(sample_1) / mean(sample_2) - 1

  Note - if mean_2 is too close to zero, then the confidence interval
  will be infinite. This happens when the confidence interval of mean_2 on it's
  own would cover 0.

  In addition to Fieller's theorem, we also make the assumption that the metric
  being analysed must always be >= 0.0. If this is not the case, it is difficult
  to interpret the relative difference. With this assumption in place, we can
  enforce that the relative difference is never lower than -1 (a -100%
  change).

  The confidence interval is constructed to match the alternative hypothesis
  being tested. So if the alternative is "less", then the lower bound will be
  -1, while if the alternative is "greater" then the upper bound will be
  infinity. If the alternative is "two-sided" then a regular confidence
  interval is constructed.

  References:
    [1] https://arxiv.org/pdf/0710.2024.pdf
    [2]
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=880fd73d65c9be84210644772d1779347a37529f

  Args:
    mean_1: The mean of the first sample.
    mean_2: The mean of the second sample.
    standard_error_1: The standard error on mean_1.
    standard_error_2: The standard error on mean_2.
    corr: The correlation between the first and second sample.
    degrees_of_freedom: The degrees of freedom of the estimates.
    alternative: The alternative hypothesis to test, one of ['two-sided',
      'greater', 'less']
    alpha: The alpha level of the confidence interval, defaults to 0.05.

  Returns:
    A tuple containing the lower and upper bounds, (lower, upper).

  Raises:
    ValueError: If either of the means are negative, or if the alternative is
      not one of ['two-sided', 'greater', 'less'].
  """
  if mean_1 < 0.0 or mean_2 < 0.0:
    raise ValueError(
        "Both mean_1 and mean_2 must be >= 0.0, "
        f"but got {mean_1 = } and {mean_2 = }."
    )

  var_1 = standard_error_1**2
  var_2 = standard_error_2**2
  var_12 = standard_error_1 * standard_error_2 * corr

  t_exclusive_sq = mean_2**2 / var_2
  t_complete_sq = (
      mean_1**2 * var_2 - 2 * mean_2 * mean_1 * var_12 + mean_2**2 * var_1
  ) / (var_1 * var_2 - var_12**2)

  if alternative == "two-sided":
    critical_t_value = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha / 2.0)
  elif alternative == "greater":
    critical_t_value = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha)
  elif alternative == "less":
    critical_t_value = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha)
  else:
    raise ValueError(
        "Alternative must be one of ['two-sided', 'greater', 'less']"
    )

  adjusted_12 = mean_1 * mean_2 - critical_t_value**2 * var_12
  adjusted_1 = mean_1**2 - critical_t_value**2 * var_1
  adjusted_2 = mean_2**2 - critical_t_value**2 * var_2

  if t_complete_sq <= critical_t_value**2:
    # The confidence interval is completely unbounded
    ratio_lower_bound = 0.0
    ratio_upper_bound = np.inf
  else:
    # The lower bound for a ratio can never be less than 0 assuming both
    # values are positive.
    ratio_lower_bound = np.clip(
        (adjusted_12 - np.sqrt(adjusted_12**2 - adjusted_1 * adjusted_2))
        / adjusted_2,
        a_min=0.0,
        a_max=None,
    )

    if t_exclusive_sq <= critical_t_value**2:
      # The confidence interval is partially unbounded, meaning the upper bound
      # is infinite but there is still a finite lower bound.
      ratio_upper_bound = np.inf
    else:
      ratio_upper_bound = (
          adjusted_12 + np.sqrt(adjusted_12**2 - adjusted_1 * adjusted_2)
      ) / adjusted_2

  lower_bound = ratio_lower_bound - 1.0
  upper_bound = ratio_upper_bound - 1.0

  if alternative == "greater":
    upper_bound = np.inf
  elif alternative == "less":
    lower_bound = -1.0

  return lower_bound, upper_bound


def _one_sample_standard_error(
    standard_deviation: npt.ArrayLike, sample_size: npt.ArrayLike
) -> npt.ArrayLike:
  """Returns the standard error on the mean.

  Args:
    standard_deviation: The standard deviation of the sample.
    sample_size: The number of observations in the sample.
  """
  return standard_deviation / np.sqrt(sample_size)


def _one_sample_degrees_of_freedom(sample_size: npt.ArrayLike) -> npt.ArrayLike:
  """Returns the degrees of freedom for the estimate of the mean of a sample.

  Args:
    sample_size: The number of observations in the sample.
  """
  return sample_size - 1


def _two_sample_standard_errors(
    standard_deviation_1: npt.ArrayLike,
    standard_deviation_2: npt.ArrayLike,
    sample_size_1: npt.ArrayLike,
    sample_size_2: npt.ArrayLike,
    equal_var: bool,
) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
  """Calculate the standard errors needed for a two sample t-test.

  Returns the standard error on the difference of the means, as well as
  the standard error on the mean of each sample separately. The former
  is needed for the t-test, while the latter is needed for the confidence
  interval calculations.

  Args:
    standard_deviation_1: The standard deviation of the first sample.
    standard_deviation_2: The standard deviation of the second sample.
    sample_size_1: The number of observations in the first sample.
    sample_size_2: The number of observations in the second sample.
    equal_var: Should the test assume both samples have equal variances?

  Returns:
    The standard error on the difference of means, the standard error of the
    mean of sample 1 and the standard error of the mean of sample 2.
  """
  if equal_var:
    pooled_var = (
        (sample_size_1 - 1) * standard_deviation_1**2
        + (sample_size_2 - 1) * standard_deviation_2**2
    ) / (sample_size_1 + sample_size_2 - 2)
    standard_error_difference = np.sqrt(
        pooled_var / sample_size_1 + pooled_var / sample_size_2
    )
    standard_error_1 = np.sqrt(pooled_var / sample_size_1)
    standard_error_2 = np.sqrt(pooled_var / sample_size_2)
  else:
    standard_error_difference = np.sqrt(
        standard_deviation_1**2 / sample_size_1
        + standard_deviation_2**2 / sample_size_2
    )
    standard_error_1 = np.sqrt(standard_deviation_1**2 / sample_size_1)
    standard_error_2 = np.sqrt(standard_deviation_2**2 / sample_size_2)

  return standard_error_difference, standard_error_1, standard_error_2


def _two_sample_degrees_of_freedom(
    standard_deviation_1: npt.ArrayLike,
    standard_deviation_2: npt.ArrayLike,
    sample_size_1: npt.ArrayLike,
    sample_size_2: npt.ArrayLike,
    equal_var: bool,
) -> npt.ArrayLike:
  """Returns the degrees of freedom for the difference of two means.

  For equal_var=True, uses the regular degrees of freedom from the students
  t-test. For equal_var=False, uses the Welch–Satterthwaite equation.

  Args:
    standard_deviation_1: The standard deviation of the first sample.
    standard_deviation_2: The standard deviation of the second sample.
    sample_size_1: The number of observations in the first sample.
    sample_size_2: The number of observations in the second sample.
    equal_var: Should the test assume both samples have equal variances?
  """
  if equal_var:
    return sample_size_1 + sample_size_2 - 2
  else:
    return (
        standard_deviation_1**2 / sample_size_1
        + standard_deviation_2**2 / sample_size_2
    ) ** 2 / (
        (standard_deviation_1**2 / sample_size_1) ** 2 / (sample_size_1 - 1)
        + (standard_deviation_2**2 / sample_size_2) ** 2 / (sample_size_2 - 1)
    )


def _ratio_mean_and_variance(
    mean_1: float, mean_2: float, var_1: float, var_2: float, cov_12: float
) -> tuple[float, float]:
  """Calculates the mean and variance of the ratio of sample 1 and 2."""

  ratio_mean = mean_1 / mean_2
  ratio_var = ratio_mean**2 * (
      var_1 / mean_1**2 + var_2 / mean_2**2 - 2 * cov_12 / (mean_1 * mean_2)
  )
  return ratio_mean, ratio_var


def _difference_of_ratios_mean_and_variance(
    mean_vector: np.ndarray, cov: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
  """Calculates the mean and variance of the difference of two ratios."""

  mean_1, var_1 = _ratio_mean_and_variance(
      mean_1=mean_vector[0],
      mean_2=mean_vector[1],
      var_1=cov[0, 0],
      var_2=cov[1, 1],
      cov_12=cov[0, 1],
  )
  mean_2, var_2 = _ratio_mean_and_variance(
      mean_1=mean_vector[2],
      mean_2=mean_vector[3],
      var_1=cov[2, 2],
      var_2=cov[3, 3],
      cov_12=cov[2, 3],
  )
  grad = np.array([
      1.0 / mean_vector[1],
      -mean_vector[0] / mean_vector[1] ** 2,
      -1.0 / mean_vector[3],
      mean_vector[2] / mean_vector[3] ** 2,
  ])
  difference_var = grad.T @ cov @ grad
  difference_mean = mean_1 - mean_2

  corr_12 = (var_1 + var_2 - difference_var) / (2 * np.sqrt(var_1 * var_2))

  return difference_mean, difference_var, mean_1, mean_2, var_1, var_2, corr_12


def yuens_t_test_paired(
    values1: np.ndarray,
    values2: np.ndarray,
    *,
    trimming_quantile: float,
    denom_values1: np.ndarray | None = None,
    denom_values2: np.ndarray | None = None,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTestResults:
  """Performs Yuen's paired t-test.

  The paired test essentially performs a 1-sample test on the differences
  between values1 and values2. Since this is a paired test, both values1 and
  values2 must have the same length.

  Yuens t-test is the equivalent of a paired t-test but for trimmed means.

  Depending on the alternative selected, this will test a different null
  and alternative hypothesis:

    alternative = "two-sided"
    H0: mean(values1) - mean(values2) = 0
    HA: mean(values1) - mean(values2) =/= 0

    alternative = "less"
    H0: mean(values1) - mean(values2) >= 0
    HA: mean(values1) - mean(values2) < 0

    alternative = "greater"
    H0: mean(values1) - mean(values2) <= 0
    HA: mean(values1) - mean(values2) > 0

  References:
    https://www.jstor.org/stable/2334550

  Args:
    values1: The first sample of data for the test.
    values2: The second sample of data for the test.
    trimming_quantile: The quantile to trim from each side of the values arrays.
    denom_values1: If analysing a ratio metric, then this should contain the
      first sample of denominator values, and values1 will be used as the
      numerator values. Defaults to None.
    denom_values2: If analysing a ratio metric, then this should contain the
      second sample of denominator values, and values2 will be used as the
      numerator values. Defaults to None.
    alternative: The alternative hypothesis to test, one of ['two-sided',
      'greater', 'less']
    alpha: The false positive rate of the test, defaults to 0.05. If the p-value
      is less than alpha, it will be statistically significant.

  Returns:
    The statistical test results.

  Raises:
    ValueError: If values1 and values2 do not have the same length.
  """
  if len(values1) != len(values2):
    raise ValueError("Values1 and values2 must have the same lengths.")

  is_ratio_metric = False
  if denom_values1 is not None and denom_values2 is not None:
    is_ratio_metric = True
  elif denom_values1 is not None or denom_values2 is not None:
    raise ValueError(
        "Must pass either both or neither denom_values1 and denom_values2."
    )

  if is_ratio_metric:
    stacked_values = np.stack(
        [values1 - values2, values1, denom_values1, values2, denom_values2]
    )
    trimmed_values = TrimmedArray(stacked_values, trimming_quantile)
    trimmed_means = trimmed_values.mean()
    trimmed_cov = trimmed_values.cov(ddof=1)

    (
        absolute_difference,
        absolute_difference_var,
        mean_1,
        mean_2,
        var_1,
        var_2,
        corr,
    ) = _difference_of_ratios_mean_and_variance(
        trimmed_means[1:], trimmed_cov[1:, 1:]
    )

    standard_error_1 = _one_sample_standard_error(
        np.sqrt(var_1), len(trimmed_values)
    )
    standard_error_2 = _one_sample_standard_error(
        np.sqrt(var_2), len(trimmed_values)
    )
    absolute_difference_standard_error = _one_sample_standard_error(
        np.sqrt(absolute_difference_var), len(trimmed_values)
    )
    sample_size = len(trimmed_values)
    degrees_of_freedom = _one_sample_degrees_of_freedom(sample_size)
  else:
    stacked_values = np.stack([
        values1 - values2,
        values1,
        values2,
    ])
    trimmed_values = TrimmedArray(stacked_values, trimming_quantile)
    trimmed_means = trimmed_values.mean()
    trimmed_cov = trimmed_values.cov(ddof=1)
    trimmed_standard_error = _one_sample_standard_error(
        trimmed_values.std(ddof=1), len(trimmed_values)
    )

    _, mean_1, mean_2 = trimmed_means
    _, standard_error_1, standard_error_2 = trimmed_standard_error
    sample_size = len(trimmed_values)
    degrees_of_freedom = _one_sample_degrees_of_freedom(sample_size)
    corr = trimmed_cov[1, 2] / np.sqrt(trimmed_cov[1, 1] * trimmed_cov[2, 2])
    absolute_difference_standard_error = trimmed_standard_error[0]
    absolute_difference = trimmed_means[0]

  statistic, p_value = _ttest_from_stats(
      point_estimate=absolute_difference,
      standard_error=absolute_difference_standard_error,
      sample_size=sample_size,
      degrees_of_freedom=degrees_of_freedom,
      alternative=alternative,
  )
  is_significant = p_value < alpha

  absolute_difference_lb, absolute_difference_ub = (
      _absolute_difference_confidence_interval(
          mean_1=mean_1,
          mean_2=mean_2,
          standard_error_1=standard_error_1,
          standard_error_2=standard_error_2,
          corr=corr,
          degrees_of_freedom=degrees_of_freedom,
          alpha=alpha,
          alternative=alternative,
      )
  )

  # We only calculate the relative lift for metrics that are positive.
  # If not, it's hard to interpret it.
  if (mean_1 >= 0.0) and (mean_2 >= 0.0):
    relative_difference = mean_1 / mean_2 - 1.0
    relative_difference_lb, relative_difference_ub = (
        _relative_difference_confidence_interval(
            mean_1=mean_1,
            mean_2=mean_2,
            standard_error_1=standard_error_1,
            standard_error_2=standard_error_2,
            corr=corr,
            degrees_of_freedom=degrees_of_freedom,
            alpha=alpha,
            alternative=alternative,
        )
    )
  else:
    relative_difference = None
    relative_difference_lb = None
    relative_difference_ub = None

  return StatisticalTestResults(
      statistic=statistic,
      p_value=p_value,
      alpha=alpha,
      is_significant=is_significant,
      absolute_difference=absolute_difference,
      absolute_difference_lower_bound=absolute_difference_lb,
      absolute_difference_upper_bound=absolute_difference_ub,
      relative_difference=relative_difference,
      relative_difference_lower_bound=relative_difference_lb,
      relative_difference_upper_bound=relative_difference_ub,
      standard_error=absolute_difference_standard_error,
      sample_size=sample_size,
      degrees_of_freedom=degrees_of_freedom,
      control_average=mean_2,
  )


def yuens_t_test_ind(
    values1: np.ndarray,
    values2: np.ndarray,
    *,
    trimming_quantile: float,
    denom_values1: np.ndarray | None = None,
    denom_values2: np.ndarray | None = None,
    equal_var: bool = False,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTestResults:
  """Performs Yuen's two-sample t-test.

  Yuen's t-test is the equivalent of a two-sample t-test but for trimmed means.
  If equal_var is set to True, then it performs a standard Yuen's two sample
  t-test. If it's set to False it performs a Yuen - Welch's t-test. We recommend
  using equal_var = False in most cases, as the assumption that the variances
  are equal between the two samples may not hold in general.

  Depending on the alternative selected, this will test a different null
  and alternative hypothesis:

    alternative = "two-sided"
    H0: mean(values1) - mean(values2) = 0
    HA: mean(values1) - mean(values2) =/= 0

    alternative = "less"
    H0: mean(values1) - mean(values2) >= 0
    HA: mean(values1) - mean(values2) < 0

    alternative = "greater"
    H0: mean(values1) - mean(values2) <= 0
    HA: mean(values1) - mean(values2) > 0

  If denom_values1 and denom_values2 are specified, then the delta method
  is used to compare two ratio metrics. In this case the null and alternative
  hypotheses are changed to compare the ratios. For example, for the two-sample
  alternative, the null and alternative hypotheses are:

    H0: (
        mean(values1) / mean(denom_values1)
        - mean(values2) / mean(denom_values2)
      ) = 0
    HA: (
        mean(values1) / mean(denom_values1)
        - mean(values2) / mean(denom_values2)
      ) =/= 0

  References:
    https://www.jstor.org/stable/40063824

  Args:
    values1: The first sample of data for the test.
    values2: The second sample of data for the test.
    trimming_quantile: The quantile to trim from each side of the values arrays.
    denom_values1: If analysing a ratio metric, then this should contain the
      first sample of denominator values, and values1 will be used as the
      numerator values. Defaults to None.
    denom_values2: If analysing a ratio metric, then this should contain the
      second sample of denominator values, and values2 will be used as the
      numerator values. Defaults to None.
    equal_var: Should the test assume both samples have equal variances?
    alternative: The alternative hypothesis to test, one of ['two-sided',
      'greater', 'less']
    alpha: The false positive rate of the test, defaults to 0.05. If the p-value
      is less than alpha, it will be statistically significant.

  Returns:
    The statistical test results.
  """
  is_ratio_metric = False
  if denom_values1 is not None and denom_values2 is not None:
    is_ratio_metric = True
  elif denom_values1 is not None or denom_values2 is not None:
    raise ValueError(
        "Must pass either both or neither denom_values1 and denom_values2."
    )

  if is_ratio_metric:
    stacked_values1 = np.stack([values1, denom_values1])
    stacked_values2 = np.stack([values2, denom_values2])
    trimmed_values1 = TrimmedArray(stacked_values1, trimming_quantile)
    trimmed_values2 = TrimmedArray(stacked_values2, trimming_quantile)
    trimmed_means_1 = trimmed_values1.mean()
    trimmed_cov_matrix_1 = trimmed_values1.cov(ddof=1)
    trimmed_means_2 = trimmed_values2.mean()
    trimmed_cov_matrix_2 = trimmed_values2.cov(ddof=1)

    mean1, var1 = _ratio_mean_and_variance(
        mean_1=trimmed_means_1[0],
        mean_2=trimmed_means_1[1],
        var_1=trimmed_cov_matrix_1[0, 0],
        var_2=trimmed_cov_matrix_1[1, 1],
        cov_12=trimmed_cov_matrix_1[0, 1],
    )
    mean2, var2 = _ratio_mean_and_variance(
        mean_1=trimmed_means_2[0],
        mean_2=trimmed_means_2[1],
        var_1=trimmed_cov_matrix_2[0, 0],
        var_2=trimmed_cov_matrix_2[1, 1],
        cov_12=trimmed_cov_matrix_2[0, 1],
    )
    n1 = len(trimmed_values1)
    n2 = len(trimmed_values2)
  else:
    trimmed_values1 = TrimmedArray(values1, trimming_quantile)
    trimmed_values2 = TrimmedArray(values2, trimming_quantile)

    mean1 = trimmed_values1.mean()
    mean2 = trimmed_values2.mean()
    var1 = trimmed_values1.var(ddof=1)
    var2 = trimmed_values2.var(ddof=1)
    n1 = len(trimmed_values1)
    n2 = len(trimmed_values2)

  standard_errors = _two_sample_standard_errors(
      standard_deviation_1=np.sqrt(var1),
      standard_deviation_2=np.sqrt(var2),
      sample_size_1=n1,
      sample_size_2=n2,
      equal_var=equal_var,
  )
  standard_error, standard_error_1, standard_error_2 = standard_errors
  degrees_of_freedom = _two_sample_degrees_of_freedom(
      standard_deviation_1=np.sqrt(var1),
      standard_deviation_2=np.sqrt(var2),
      sample_size_1=n1,
      sample_size_2=n2,
      equal_var=equal_var,
  )

  absolute_difference = mean1 - mean2

  statistic, p_value = _ttest_from_stats(
      point_estimate=absolute_difference,
      standard_error=standard_error,
      sample_size=n1 + n2,
      degrees_of_freedom=degrees_of_freedom,
      alternative=alternative,
  )
  is_significant = p_value < alpha

  absolute_difference_lb, absolute_difference_ub = (
      _absolute_difference_confidence_interval(
          mean_1=mean1,
          mean_2=mean2,
          standard_error_1=standard_error_1,
          standard_error_2=standard_error_2,
          corr=0.0,
          degrees_of_freedom=degrees_of_freedom,
          alpha=alpha,
          alternative=alternative,
      )
  )

  # We only calculate the relative lift for metrics that are positive.
  # If not, it's hard to interpret it.
  if (mean1 >= 0.0) and (mean2 >= 0.0):
    relative_difference = mean1 / mean2 - 1.0
    relative_difference_lb, relative_difference_ub = (
        _relative_difference_confidence_interval(
            mean_1=mean1,
            mean_2=mean2,
            standard_error_1=standard_error_1,
            standard_error_2=standard_error_2,
            corr=0.0,
            degrees_of_freedom=degrees_of_freedom,
            alpha=alpha,
            alternative=alternative,
        )
    )
  else:
    relative_difference = None
    relative_difference_lb = None
    relative_difference_ub = None

  return StatisticalTestResults(
      statistic=statistic,
      p_value=p_value,
      alpha=alpha,
      is_significant=is_significant,
      absolute_difference=absolute_difference,
      absolute_difference_lower_bound=absolute_difference_lb,
      absolute_difference_upper_bound=absolute_difference_ub,
      relative_difference=relative_difference,
      relative_difference_lower_bound=relative_difference_lb,
      relative_difference_upper_bound=relative_difference_ub,
      standard_error=standard_error,
      sample_size=n1 + n2,
      degrees_of_freedom=degrees_of_freedom,
      control_average=mean2,
  )


def calculate_minimum_detectable_effect_from_stats(
    standard_error: float,
    degrees_of_freedom: int | float,
    alternative: str,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
  """Calculates the minimum detectable effect of an experiment.

  The minimum detectable effect is the smallest true effect size that would
  return a statistically significant result, at the specificed alpha level
  and with the specified alternative hypothesis, [power]% of the time.

  The returned minimum detectable effect is always positive, even if the
  alternative hypothesis is "less".

  Args:
    standard_error: The standard error of the test statistic.
    degrees_of_freedom: The degrees of freedom of the test statistic.
    alternative: The alternative hypothesis being tested, one of ['two-sided',
      'greater', 'less'].
    power: The desired statistical power, as a fraction. Defaults to 0.8, which
      would mean a power of 80%.
    alpha: The alpha level of the test, defaults to 0.05.

  Returns:
    The minimum detectable absolute effect size.

  Raises:
     ValueError: If the alternative is not one of ['two-sided', 'greater',
      'less'].
  """
  if alternative == "two-sided":
    t_alpha = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha / 2.0)
  elif alternative in ["greater", "less"]:
    t_alpha = stats.t.ppf(df=degrees_of_freedom, q=1.0 - alpha)
  else:
    raise ValueError(
        "Alternative must be one of ['two-sided', 'greater', 'less']"
    )

  t_power = stats.t.ppf(power, degrees_of_freedom)

  return standard_error * (t_alpha + t_power)


def yuens_t_test_ind_minimum_detectable_effect(
    *,
    standard_deviation: npt.ArrayLike,
    sample_size: npt.ArrayLike,
    alternative: str = "two-sided",
    power: float = 0.8,
    alpha: float = 0.05,
) -> npt.ArrayLike:
  """Calculates the minimum detectable effect of an independent two sample test.

  This calculates the minimum detectable effect under the assumption that
  the samples will be ramdomly assigned to control and treatment with a 50/50
  split.

  The minimum detectable effect is the smallest true effect size that would
  return a statistically significant result, at the specificed alpha level
  and with the specified alternative hypothesis, [power]% of the time.

  The returned minimum detectable effect is always positive, even if the
  alternative hypothesis is "less".

  There is no option to select unequal variances here, because even though it's
  a two sample test, when we are performing the power calculation we typically
  only have a single sample to estimate the variances from, so we set the
  variance to be identical in both samples. This means that regardless of
  whether we assume equal variances or not, we get the same results.

  Args:
    standard_deviation: The bias corrected standard deviation of the sample of
      data on which the experiment would be carried out. In practice this is
      typically a historical sample of data. If trimming, this should be the
      winsorized standard deviation.
    sample_size: The number of samples for the experiment. This is the total
      number of samples, and it's assumed that they would be split 50/50.
    alternative: The alternative hypothesis to test, one of ['two-sided',
      'greater', 'less']
    power: The desired statistical power, as a fraction. Defaults to 0.8, which
      would mean a power of 80%.
    alpha: The false positive rate of the test, defaults to 0.05. If the p-value
      is less than alpha, it will be statistically significant.

  Returns:
    The minimum detectable absolute effect size.
  """
  sample_size_per_variant = 0.5 * sample_size

  standard_error, _, _ = _two_sample_standard_errors(
      standard_deviation_1=standard_deviation,
      standard_deviation_2=standard_deviation,
      sample_size_1=sample_size_per_variant,
      sample_size_2=sample_size_per_variant,
      equal_var=True,
  )
  degrees_of_freedom = _two_sample_degrees_of_freedom(
      standard_deviation_1=standard_deviation,
      standard_deviation_2=standard_deviation,
      sample_size_1=sample_size_per_variant,
      sample_size_2=sample_size_per_variant,
      equal_var=True,
  )

  minimum_detectable_effect = calculate_minimum_detectable_effect_from_stats(
      standard_error=standard_error,
      degrees_of_freedom=degrees_of_freedom,
      alternative=alternative,
      power=power,
      alpha=alpha,
  )

  return minimum_detectable_effect


def yuens_t_test_paired_minimum_detectable_effect(
    *,
    difference_standard_deviation: npt.ArrayLike,
    sample_size: npt.ArrayLike,
    alternative: str = "two-sided",
    power: float = 0.8,
    alpha: float = 0.05,
) -> npt.ArrayLike:
  """Calculates the minimum detectable effect of a paired two sample test.

  The difference_standard_deviation should be calculated by estimating the

  This calculates the minimum detectable effect under the assumption that
  when running the experiment, the samples will be ramdomly assigned with a
  50/50 split, with 50% of the samples having the treatment applied in period 1,
  and the other 50% having the treatment applied in period 2. For the analysis,
  the samples would be aligned, so that the paired test is comparing all of the
  samples when they were treated with all of the samples when they were not
  treated. It also assumes that sample 1 and sample 2 would be de-meaned before
  analyzing them, so that the mean of period 1 is the same as the mean of period
  2.

  The minimum detectable effect is the smallest true effect size that would
  return a statistically significant result, at the specificed alpha level
  and with the specified alternative hypothesis, [power]% of the time.

  The returned minimum detectable effect is always positive, even if the
  alternative hypothesis is "less".

  Args:
    difference_standard_deviation: The bias corrected standard deviation of the
      paired difference between the two samples of data. If trimming, this
      should be the winsorized standard deviation.
    sample_size: The total number of samples for the experiment.
    alternative: The alternative hypothesis to test, one of ['two-sided',
      'greater', 'less']
    power: The desired statistical power, as a fraction. Defaults to 0.8, which
      would mean a power of 80%.
    alpha: The false positive rate of the test, defaults to 0.05. If the p-value
      is less than alpha, it will be statistically significant.

  Returns:
    The minimum detectable absolute effect size.
  """
  standard_error = _one_sample_standard_error(
      difference_standard_deviation, sample_size
  )
  degrees_of_freedom = _one_sample_degrees_of_freedom(sample_size)

  return calculate_minimum_detectable_effect_from_stats(
      standard_error=standard_error,
      degrees_of_freedom=degrees_of_freedom,
      alternative=alternative,
      power=power,
      alpha=alpha,
  )
