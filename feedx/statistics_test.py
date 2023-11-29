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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import stats

from feedx import statistics


class TrimmedArrayTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.input_array = np.array([5, 7, 8.5, 1, 9.5, 12, 20, 1, 2, 3])
    self.quantile = 0.25
    self.expected_trimmed_array = np.array([[5, 7, 8.5, 9.5, 2, 3]])
    self.expected_winsorized_array = np.array(
        [2, 2, 2, 3, 5, 7, 8.5, 9.5, 9.5, 9.5]
    )

    self.input_array_2 = np.array([3, 6, 1.0, 0.4, 2.0, 1.0, 9.0, 0.5, 0.2, 1])

  @parameterized.parameters([2.0, 0.5, -0.01])
  def test_raises_exception_if_quantile_not_in_expected_range(self, quantile):
    with self.assertRaises(ValueError):
      statistics.TrimmedArray(self.input_array, quantile=quantile)

  def test_the_number_of_elements_to_trim_is_calculated_from_the_quantile(self):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=self.quantile
    )

    self.assertEqual(trimmed_array.n_trim, 2)
    np.testing.assert_array_equal(
        trimmed_array.values_trimmed, self.expected_trimmed_array
    )

  def test_mean_is_the_trimmed_mean(self):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=self.quantile
    )

    expected_trimmed_mean = np.mean(self.expected_trimmed_array)

    self.assertAlmostEqual(trimmed_array.mean(), expected_trimmed_mean)

  def test_winsorized_mean_is_correct(self):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=self.quantile
    )

    expected_winsorized_mean = np.mean(self.expected_winsorized_array)

    self.assertAlmostEqual(
        trimmed_array.winsorized_mean(), expected_winsorized_mean
    )

  @parameterized.parameters([0, 1])
  def test_variance_and_standard_deviation_are_winsorized(self, ddof):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=self.quantile
    )

    winsorized_mean = np.mean(self.expected_winsorized_array)
    winsorized_resid_sq = (
        self.expected_winsorized_array - winsorized_mean
    ) ** 2
    degrees_of_freedom = len(self.expected_trimmed_array.flatten()) - ddof
    expected_winsorized_var = np.sum(winsorized_resid_sq) / degrees_of_freedom

    self.assertAlmostEqual(trimmed_array.var(ddof), expected_winsorized_var)
    self.assertAlmostEqual(
        trimmed_array.std(ddof), np.sqrt(expected_winsorized_var)
    )

  def test_len_returns_trimmed_length(self):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=self.quantile
    )

    expected_length = len(self.expected_trimmed_array.flatten())

    self.assertLen(trimmed_array, expected_length)

  def test_when_no_items_are_trimmed_mean_matches_standard_numpy(self):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=0.01
    )  # quantile=0.01 is small enough that no trimming happens.

    self.assertEqual(trimmed_array.n_trim, 0)
    self.assertAlmostEqual(trimmed_array.mean(), self.input_array.mean())

  @parameterized.parameters([0, 1])
  def test_when_no_items_are_trimmed_var_matches_standard_numpy(self, ddof):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=0.01
    )  # quantile=0.01 is small enough that no trimming happens.

    self.assertEqual(trimmed_array.n_trim, 0)
    self.assertAlmostEqual(
        trimmed_array.var(ddof=ddof), self.input_array.var(ddof=ddof)
    )

  @parameterized.parameters([0, 1])
  def test_when_no_items_are_trimmed_std_matches_standard_numpy(self, ddof):
    trimmed_array = statistics.TrimmedArray(
        self.input_array, quantile=0.01
    )  # quantile=0.01 is small enough that no trimming happens.

    self.assertEqual(trimmed_array.n_trim, 0)
    self.assertAlmostEqual(
        trimmed_array.std(ddof=ddof), self.input_array.std(ddof=ddof)
    )

  def test_2d_trimmed_array_calculates_covariance_matrix(self):
    input_arrat_2d = np.stack([
        self.input_array,
        self.input_array_2,
    ])

    trimmed_array = statistics.TrimmedArray(
        input_arrat_2d, quantile=self.quantile
    )
    cov = trimmed_array.cov()

    self.assertEqual(cov.shape, (2, 2))

  def test_when_no_items_are_trimmed_2d_mean_matches_standard_numpy(self):
    input_arrat_2d = np.stack([
        self.input_array,
        self.input_array_2,
    ])

    trimmed_array = statistics.TrimmedArray(
        input_arrat_2d, quantile=0.01
    )  # quantile=0.01 is small enough that no trimming happens.

    self.assertEqual(trimmed_array.n_trim, 0)
    np.testing.assert_allclose(
        trimmed_array.mean(), input_arrat_2d.mean(axis=1)
    )

  @parameterized.parameters([0, 1])
  def test_when_no_items_are_trimmed_2d_var_matches_standard_numpy(self, ddof):
    input_arrat_2d = np.stack([
        self.input_array,
        self.input_array_2,
    ])

    trimmed_array = statistics.TrimmedArray(
        input_arrat_2d, quantile=0.01
    )  # quantile=0.01 is small enough that no trimming happens.

    self.assertEqual(trimmed_array.n_trim, 0)
    np.testing.assert_allclose(
        trimmed_array.var(ddof=ddof), input_arrat_2d.var(ddof=ddof, axis=1)
    )

  @parameterized.parameters([0, 1])
  def test_when_no_items_are_trimmed_2d_std_matches_standard_numpy(self, ddof):
    input_arrat_2d = np.stack([
        self.input_array,
        self.input_array_2,
    ])

    trimmed_array = statistics.TrimmedArray(
        input_arrat_2d, quantile=0.01
    )  # quantile=0.01 is small enough that no trimming happens.

    self.assertEqual(trimmed_array.n_trim, 0)
    np.testing.assert_allclose(
        trimmed_array.std(ddof=ddof), input_arrat_2d.std(ddof=ddof, axis=1)
    )

  @parameterized.parameters([0, 1])
  def test_when_no_items_are_trimmed_2d_cov_matches_standard_numpy(self, ddof):
    input_arrat_2d = np.stack([
        self.input_array,
        self.input_array_2,
    ])

    trimmed_array = statistics.TrimmedArray(
        input_arrat_2d, quantile=0.01
    )  # quantile=0.01 is small enough that no trimming happens.

    self.assertEqual(trimmed_array.n_trim, 0)
    np.testing.assert_allclose(
        trimmed_array.cov(ddof=ddof), np.cov(input_arrat_2d, ddof=ddof)
    )

  @parameterized.parameters([0.0, 0.1, 0.2, 0.3])
  def test_variance_of_difference_can_be_estimated_from_covariance_of_individuals(
      self, quantile
  ):
    # This is something that is important for the paired Yuen's t-test

    # To make this work we trim based on the difference, but then calculate
    # the covariance of the original two arrays with that trimming.
    input_arrat_2d = np.stack([
        self.input_array - self.input_array_2,
        self.input_array,
        self.input_array_2,
    ])

    trimmed_array = statistics.TrimmedArray(input_arrat_2d, quantile=quantile)
    cov = trimmed_array.cov()
    delta_var = cov[0, 0]
    var_1 = cov[1, 1]
    var_2 = cov[2, 2]
    cov_12 = cov[1, 2]

    # Delta method:
    # Var[X - Y] = Var[X] + Var[Y] - 2 * Covar[X,Y]
    delta_var_est_from_covar = var_1 + var_2 - 2 * cov_12

    self.assertAlmostEqual(delta_var, delta_var_est_from_covar)


class CupedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.metric = np.array([1, 1, 2, 3, 5, 7, 8.5, 9.5, 12, 20])
    self.covariate = np.array([1, 3, 1, 4, 6, 4, 9.5, 10.5, 12, 15])
    self.corr = np.corrcoef(self.covariate, self.metric)[1, 0]  # 0.9399

  def test_variance_is_reduced_after_cuped_adjustment(self):
    cuped_metric = statistics.apply_cuped_adjustment(
        self.metric, self.covariate
    )

    self.assertLess(cuped_metric.var(), self.metric.var())
    self.assertAlmostEqual(
        cuped_metric.var(), self.metric.var() * (1.0 - self.corr**2)
    )

  def test_mean_is_unchanged_after_cuped_adjustment(self):
    cuped_metric = statistics.apply_cuped_adjustment(
        self.metric, self.covariate
    )

    self.assertAlmostEqual(cuped_metric.mean(), self.metric.mean())


class YuensTTestIndTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.sample_1 = np.array([1, 1, 2, 3, 5, 7, 8.5, 9.5, 12, 20])
    self.sample_2 = np.array([3, 6, 3, 2, 3, 4, 5.0, 0.2, 0.1, 0.5, 1, 2, 1])
    self.quantile = 0.25

    self.trimmed_sample_1 = statistics.TrimmedArray(
        self.sample_1, self.quantile
    )
    self.trimmed_sample_2 = statistics.TrimmedArray(
        self.sample_2, self.quantile
    )

  def test_one_sample_standard_error_returns_expected_value(self):
    standard_deviation = 2.0
    sample_size = 150

    actual_standard_error = statistics._one_sample_standard_error(
        standard_deviation, sample_size
    )

    expected_standard_error = standard_deviation / np.sqrt(sample_size)
    self.assertEqual(actual_standard_error, expected_standard_error)

  @parameterized.named_parameters([
      dict(
          testcase_name="equal_var",
          equal_var=True,
          expected_standard_error_difference=0.2828850408914428,
          expected_standard_error_1=0.21384099080545574,
          expected_standard_error_2=0.18519173040795925,
      ),
      dict(
          testcase_name="unequal_var",
          equal_var=False,
          expected_standard_error_difference=0.26770630673681683,
          expected_standard_error_1=0.16329931618554522,
          expected_standard_error_2=0.21213203435596426,
      ),
  ])
  def test_two_sample_standard_errors_returns_expected_standard_errors(
      self,
      equal_var,
      expected_standard_error_difference,
      expected_standard_error_1,
      expected_standard_error_2,
  ):
    standard_deviation_1 = 2.0
    standard_deviation_2 = 3.0
    sample_size_1 = 150
    sample_size_2 = 200

    result = statistics._two_sample_standard_errors(
        standard_deviation_1=standard_deviation_1,
        standard_deviation_2=standard_deviation_2,
        sample_size_1=sample_size_1,
        sample_size_2=sample_size_2,
        equal_var=equal_var,
    )

    self.assertEqual(
        result,
        (
            expected_standard_error_difference,
            expected_standard_error_1,
            expected_standard_error_2,
        ),
    )

  def test_one_sample_degrees_of_freedom_is_sample_size_minus_1(self):
    sample_size = 150

    actual_degrees_of_freedom = statistics._one_sample_degrees_of_freedom(
        sample_size
    )

    expected_degrees_of_freedom = sample_size - 1
    self.assertEqual(actual_degrees_of_freedom, expected_degrees_of_freedom)

  @parameterized.parameters(True, False)
  def test_two_sample_standard_errors_difference_is_sum_of_squares_of_each_sample(
      self, equal_var
  ):
    standard_deviation_1 = 2.0
    standard_deviation_2 = 3.0
    sample_size_1 = 150
    sample_size_2 = 200

    result = statistics._two_sample_standard_errors(
        standard_deviation_1=standard_deviation_1,
        standard_deviation_2=standard_deviation_2,
        sample_size_1=sample_size_1,
        sample_size_2=sample_size_2,
        equal_var=equal_var,
    )

    standard_error_difference, standard_error_1, standard_error_2 = result
    self.assertAlmostEqual(
        standard_error_difference**2, standard_error_1**2 + standard_error_2**2
    )

  @parameterized.parameters([(True, 348), (False, 343.5884999843324)])
  def test_two_sample_degrees_of_freedom_is_expected_degrees_of_freedom(
      self, equal_var, expected_degrees_of_freedom
  ):
    standard_deviation_1 = 2.0
    standard_deviation_2 = 3.0
    sample_size_1 = 150
    sample_size_2 = 200

    degrees_of_freedom = statistics._two_sample_degrees_of_freedom(
        standard_deviation_1=standard_deviation_1,
        standard_deviation_2=standard_deviation_2,
        sample_size_1=sample_size_1,
        sample_size_2=sample_size_2,
        equal_var=equal_var,
    )

    self.assertAlmostEqual(degrees_of_freedom, expected_degrees_of_freedom)

  def test_absolute_difference_is_difference_in_sample_means(
      self,
  ):
    result = statistics.yuens_t_test_ind(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )

    expected_absolute_difference = (
        self.trimmed_sample_1.mean() - self.trimmed_sample_2.mean()
    )
    print(expected_absolute_difference)
    print(result.absolute_difference)

    self.assertEqual(result.absolute_difference, expected_absolute_difference)

  def test_returns_expected_sample_size(self):
    result = statistics.yuens_t_test_ind(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )

    expected_sample_size = len(self.trimmed_sample_1) + len(
        self.trimmed_sample_2
    )

    self.assertEqual(result.sample_size, expected_sample_size)

  @parameterized.product(
      equal_var=[True, False], alternative=["two-sided", "greater", "less"]
  )
  def test_with_no_trimming_matches_scipy_p_value(self, equal_var, alternative):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=0.0,
        equal_var=equal_var,
        alternative=alternative,
    )

    scipy_result = stats.ttest_ind(
        self.sample_1,
        self.sample_2,
        equal_var=equal_var,
        alternative=alternative,
    )

    self.assertAlmostEqual(result.p_value, scipy_result.pvalue)

  @parameterized.product(
      equal_var=[True, False], alternative=["two-sided", "greater", "less"]
  )
  def test_with_no_trimming_matches_scipy_statistic(
      self, equal_var, alternative
  ):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=0.0,
        equal_var=equal_var,
        alternative=alternative,
    )

    scipy_result = stats.ttest_ind(
        self.sample_1,
        self.sample_2,
        equal_var=equal_var,
        alternative=alternative,
    )

    self.assertAlmostEqual(result.statistic, scipy_result.statistic)

  def test_defaults_to_two_sided_and_unequal_var(self):
    result_two_sided_unequal_var = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="two-sided",
        equal_var=False,
    )
    result_default = statistics.yuens_t_test_ind(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )

    self.assertEqual(result_two_sided_unequal_var, result_default)

  @parameterized.parameters([0.01, 0.05, 0.1])
  def test_sets_alpha_to_input_alpha(self, alpha):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alpha=alpha,
    )

    self.assertEqual(result.alpha, alpha)

  @parameterized.parameters([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
  def test_returns_significant_result_if_pvalue_less_than_alpha(self, alpha):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alpha=alpha,
    )

    pvalue_less_than_alpha = result.p_value < alpha
    self.assertEqual(result.is_significant, pvalue_less_than_alpha)

  @parameterized.named_parameters([
      {
          "testcase_name": "sample_1_negative",
          "sample_1_sign": -1.0,
          "sample_2_sign": 1.0,
      },
      {
          "testcase_name": "sample_2_negative",
          "sample_1_sign": 1.0,
          "sample_2_sign": -1.0,
      },
      {
          "testcase_name": "both_samples_negative",
          "sample_1_sign": -1.0,
          "sample_2_sign": -1.0,
      },
  ])
  def test_sets_relative_difference_to_none_if_means_are_negative(
      self, sample_1_sign, sample_2_sign
  ):
    result = statistics.yuens_t_test_ind(
        self.sample_1 * sample_1_sign,
        self.sample_2 * sample_2_sign,
        trimming_quantile=self.quantile,
    )

    self.assertIsNone(result.relative_difference)
    self.assertIsNone(result.relative_difference_lower_bound)
    self.assertIsNone(result.relative_difference_upper_bound)

  def test_relative_difference_lower_bound_less_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertLess(
        result.relative_difference_lower_bound, result.relative_difference
    )

  def test_relative_difference_upper_bound_greater_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertGreater(
        result.relative_difference_upper_bound, result.relative_difference
    )

  def test_absolute_difference_lower_bound_less_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertLess(
        result.absolute_difference_lower_bound, result.absolute_difference
    )

  def test_absolute_difference_upper_bound_greater_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertGreater(
        result.absolute_difference_upper_bound, result.absolute_difference
    )

  def test_alternative_greater_lower_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="greater",
    )

    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="greater",
        alpha=unadjusted_result.p_value,
    )

    self.assertAlmostEqual(result.p_value, result.alpha)
    self.assertAlmostEqual(result.absolute_difference_lower_bound, 0.0)
    self.assertAlmostEqual(result.relative_difference_lower_bound, 0.0)

  def test_alternative_less_upper_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_ind(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="less",
    )

    result = statistics.yuens_t_test_ind(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="less",
        alpha=unadjusted_result.p_value,
    )

    self.assertAlmostEqual(result.p_value, result.alpha)
    self.assertAlmostEqual(result.absolute_difference_upper_bound, 0.0)
    self.assertAlmostEqual(result.relative_difference_upper_bound, 0.0)

  def test_alternative_two_sided_lower_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="two-sided",
    )

    result = statistics.yuens_t_test_ind(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="two-sided",
        alpha=unadjusted_result.p_value,
    )

    self.assertAlmostEqual(result.p_value, result.alpha)
    self.assertAlmostEqual(result.absolute_difference_lower_bound, 0.0)
    self.assertAlmostEqual(result.relative_difference_lower_bound, 0.0)

  def test_alternative_two_sided_upper_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_ind(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="two-sided",
    )

    result = statistics.yuens_t_test_ind(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="two-sided",
        alpha=unadjusted_result.p_value,
    )

    self.assertAlmostEqual(result.p_value, result.alpha)
    self.assertAlmostEqual(result.absolute_difference_upper_bound, 0.0)
    self.assertAlmostEqual(result.relative_difference_upper_bound, 0.0)


class YuensTTestPairedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.sample_1 = np.array([1, 1, 2, 3, 5, 7, 8.5, 9.5, 12, 20])
    self.sample_2 = np.array([3, 6, 3, 2, 3, 4, 5.0, 0.2, 0.1, 0.5])
    self.delta_sample = self.sample_1 - self.sample_2
    self.quantile = 0.25

    self.trimmed_sample_1 = statistics.TrimmedArray(
        self.sample_1, self.quantile
    )
    self.trimmed_sample_2 = statistics.TrimmedArray(
        self.sample_2, self.quantile
    )
    self.trimmed_delta_sample = statistics.TrimmedArray(
        self.delta_sample, self.quantile
    )

  def test_absolute_difference_is_absolute_difference_of_means(
      self,
  ):
    result = statistics.yuens_t_test_paired(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )

    self.assertEqual(
        result.absolute_difference, self.trimmed_delta_sample.mean()
    )

  def test_tandard_error_is_correct(self):
    result = statistics.yuens_t_test_paired(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )
    expected_standard_error = self.trimmed_delta_sample.std(ddof=1) / np.sqrt(
        len(self.trimmed_delta_sample)
    )

    self.assertEqual(result.standard_error, expected_standard_error)

  def test_has_expected_sample_size(self):
    result = statistics.yuens_t_test_paired(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )
    expected_sample_size = len(self.trimmed_delta_sample)

    self.assertEqual(result.sample_size, expected_sample_size)

  def test_has_expected_degrees_of_freedom(self):
    result = statistics.yuens_t_test_paired(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )
    expected_sample_size = len(self.trimmed_delta_sample)

    self.assertEqual(result.degrees_of_freedom, expected_sample_size - 1)

  @parameterized.parameters(["two-sided", "greater", "less"])
  def test_with_no_trimming_matches_scipy_p_value(self, alternative):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=0.0,
        alternative=alternative,
    )

    scipy_result = stats.ttest_1samp(
        self.sample_1 - self.sample_2, popmean=0, alternative=alternative
    )

    self.assertAlmostEqual(scipy_result.pvalue, result.p_value)

  @parameterized.parameters(["two-sided", "greater", "less"])
  def test_with_no_trimming_matches_scipy_statistic(self, alternative):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=0.0,
        alternative=alternative,
    )

    scipy_result = stats.ttest_1samp(
        self.sample_1 - self.sample_2, popmean=0, alternative=alternative
    )

    self.assertAlmostEqual(scipy_result.statistic, result.statistic)

  def test_defaults_to_two_sided(self):
    result_two_sided = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="two-sided",
    )
    result_default = statistics.yuens_t_test_paired(
        self.sample_1, self.sample_2, trimming_quantile=self.quantile
    )

    self.assertEqual(result_two_sided, result_default)

  @parameterized.parameters([0.01, 0.05, 0.1])
  def test_sets_alpha_to_input_alpha(self, alpha):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alpha=alpha,
    )

    self.assertEqual(result.alpha, alpha)

  @parameterized.parameters([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
  def test_returns_significant_result_if_pvalue_less_than_alpha(self, alpha):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alpha=alpha,
    )

    pvalue_less_than_alpha = result.p_value < alpha
    self.assertEqual(result.is_significant, pvalue_less_than_alpha)

  @parameterized.named_parameters([
      {
          "testcase_name": "sample_1_negative",
          "sample_1_sign": -1.0,
          "sample_2_sign": 1.0,
      },
      {
          "testcase_name": "sample_2_negative",
          "sample_1_sign": 1.0,
          "sample_2_sign": -1.0,
      },
      {
          "testcase_name": "both_samples_negative",
          "sample_1_sign": -1.0,
          "sample_2_sign": -1.0,
      },
  ])
  def test_sets_relative_difference_to_none_if_means_are_negative(
      self, sample_1_sign, sample_2_sign
  ):
    result = statistics.yuens_t_test_paired(
        self.sample_1 * sample_1_sign,
        self.sample_2 * sample_2_sign,
        trimming_quantile=self.quantile,
    )

    self.assertEqual(
        [
            result.relative_difference,
            result.relative_difference_lower_bound,
            result.relative_difference_upper_bound,
        ],
        [None, None, None],
    )

  def test_relative_difference_lower_bound_less_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertLess(
        result.relative_difference_lower_bound, result.relative_difference
    )

  def test_relative_difference_upper_bound_greater_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertGreater(
        result.relative_difference_upper_bound, result.relative_difference
    )

  def test_absolute_difference_lower_bound_less_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertLess(
        result.absolute_difference_lower_bound, result.absolute_difference
    )

  def test_absolute_difference_upper_bound_greater_than_point_estimate(
      self,
  ):
    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
    )

    self.assertGreater(
        result.absolute_difference_upper_bound, result.absolute_difference
    )

  def test_alternative_greater_lower_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="greater",
    )

    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="greater",
        alpha=unadjusted_result.p_value,
    )

    actual = np.array([
        result.p_value,
        result.absolute_difference_lower_bound,
        result.relative_difference_lower_bound,
    ])
    expected = np.array([
        result.alpha,
        0.0,
        0.0,
    ])

    np.testing.assert_array_almost_equal(actual, expected)

  def test_alternative_less_upper_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_paired(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="less",
    )

    result = statistics.yuens_t_test_paired(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="less",
        alpha=unadjusted_result.p_value,
    )

    actual = np.array([
        result.p_value,
        result.absolute_difference_upper_bound,
        result.relative_difference_upper_bound,
    ])
    expected = np.array([
        result.alpha,
        0.0,
        0.0,
    ])

    np.testing.assert_array_almost_equal(actual, expected)

  def test_alternative_two_sided_lower_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="two-sided",
    )

    result = statistics.yuens_t_test_paired(
        self.sample_1,
        self.sample_2,
        trimming_quantile=self.quantile,
        alternative="two-sided",
        alpha=unadjusted_result.p_value,
    )

    actual = np.array([
        result.p_value,
        result.absolute_difference_lower_bound,
        result.relative_difference_lower_bound,
    ])
    expected = np.array([
        result.alpha,
        0.0,
        0.0,
    ])

    np.testing.assert_array_almost_equal(actual, expected)

  def test_alternative_two_sided_upper_bound_is_zero_when_p_value_equals_alpha(
      self,
  ):
    """Tests for consistency between the confidence intervals and p_values."""

    unadjusted_result = statistics.yuens_t_test_paired(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="two-sided",
    )

    result = statistics.yuens_t_test_paired(
        self.sample_2,
        self.sample_1,
        trimming_quantile=self.quantile,
        alternative="two-sided",
        alpha=unadjusted_result.p_value,
    )

    actual = np.array([
        result.p_value,
        result.absolute_difference_upper_bound,
        result.relative_difference_upper_bound,
    ])
    expected = np.array([
        result.alpha,
        0.0,
        0.0,
    ])

    np.testing.assert_array_almost_equal(actual, expected)


class RelativeDifferenceConfidenceIntervalTest(parameterized.TestCase):

  def test_alternative_two_sided_bounds_in_expected_range(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 2.0
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 / mean_2 - 1
    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="two-sided",
    )

    self.assertGreater(point_estimate, lb)
    self.assertGreater(ub, point_estimate)
    self.assertGreater(lb, -1.0)
    self.assertLess(ub, np.inf)

  def test_alternative_two_sided_when_mean_2_close_to_zero_upper_bound_is_infinite(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 0.25
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 / mean_2 - 1
    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="two-sided",
    )

    self.assertGreater(point_estimate, lb)
    self.assertGreater(ub, point_estimate)
    self.assertGreater(lb, -1.0)
    self.assertEqual(ub, np.inf)

  def test_alternative_two_sided_when_mean_1_close_to_zero_lower_bound_is_minus_1(
      self,
  ):
    mean_1 = 0.5
    mean_2 = 1.5
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 / mean_2 - 1
    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="two-sided",
    )

    self.assertGreater(point_estimate, lb)
    self.assertGreater(ub, point_estimate)
    self.assertEqual(lb, -1.0)
    self.assertGreater(ub, -1.0)
    self.assertLess(ub, np.inf)

  def test_alternative_two_sided_when_mean_2_and_mean_1_close_to_zero_lower_bound_is_minus_1_upper_bound_is_infinite(
      self,
  ):
    mean_1 = 0.5
    mean_2 = 0.25
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 / mean_2 - 1
    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="two-sided",
    )

    self.assertGreater(point_estimate, lb)
    self.assertGreater(ub, point_estimate)
    self.assertEqual(lb, -1.0)
    self.assertEqual(ub, np.inf)

  def test_alternative_greater_upper_bound_is_inf(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 2.0
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 / mean_2 - 1
    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="greater",
    )

    self.assertGreater(point_estimate, lb)
    self.assertGreater(lb, -1.0)
    self.assertEqual(ub, np.inf)

  def test_alternative_greater_when_mean_1_close_to_zero_lower_bound_is_minus_1_upper_bound_is_infinite(
      self,
  ):
    mean_1 = 0.5
    mean_2 = 1.5
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="greater",
    )

    self.assertListEqual([lb, ub], [-1.0, np.inf])

  def test_alternative_less_lower_bound_is_minus_1(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 2.0
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 / mean_2 - 1
    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="less",
    )

    self.assertGreater(point_estimate, lb)
    self.assertGreater(ub, point_estimate)
    self.assertEqual(lb, -1.0)
    self.assertLess(ub, np.inf)

  def test_alternative_less_when_mean_2_close_to_zero_lower_bound_is_minus_1_upper_bound_is_infinite(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 0.25
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    lb, ub = statistics._relative_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="less",
    )

    self.assertEqual([lb, ub], [-1.0, np.inf])


class AbsoluteDifferenceConfidenceIntervalTest(parameterized.TestCase):

  def test_calculates_expected_confidence_interval_alternative_two_sided(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 2.0
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 - mean_2
    expected_standard_error = np.sqrt(
        standard_error_1**2
        + standard_error_2**2
        - 2 * standard_error_1 * standard_error_2 * corr
    )
    expected_lb, expected_ub = stats.t.interval(
        0.95,
        df=degrees_of_freedom,
        loc=point_estimate,
        scale=expected_standard_error,
    )

    lb, ub = statistics._absolute_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="two-sided",
    )

    expected_interval = np.array([expected_lb, expected_ub])
    actual_interval = np.array([lb, ub])
    np.testing.assert_allclose(expected_interval, actual_interval)

  def test_calculates_expected_confidence_interval_alternative_greater(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 2.0
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 - mean_2
    expected_standard_error = np.sqrt(
        standard_error_1**2
        + standard_error_2**2
        - 2 * standard_error_1 * standard_error_2 * corr
    )
    expected_lb = stats.t.ppf(
        0.05,
        df=degrees_of_freedom,
        loc=point_estimate,
        scale=expected_standard_error,
    )

    lb, ub = statistics._absolute_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="greater",
    )

    expected_interval = np.array([expected_lb, np.inf])
    actual_interval = np.array([lb, ub])
    np.testing.assert_allclose(expected_interval, actual_interval)

  def test_calculates_expected_confidence_interval_alternative_less(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 2.0
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    point_estimate = mean_1 - mean_2
    expected_standard_error = np.sqrt(
        standard_error_1**2
        + standard_error_2**2
        - 2 * standard_error_1 * standard_error_2 * corr
    )
    expected_ub = stats.t.ppf(
        0.95,
        df=degrees_of_freedom,
        loc=point_estimate,
        scale=expected_standard_error,
    )

    lb, ub = statistics._absolute_difference_confidence_interval(
        mean_1=mean_1,
        mean_2=mean_2,
        standard_error_1=standard_error_1,
        standard_error_2=standard_error_2,
        corr=corr,
        degrees_of_freedom=degrees_of_freedom,
        alternative="less",
    )

    expected_interval = np.array([-np.inf, expected_ub])
    actual_interval = np.array([lb, ub])
    np.testing.assert_allclose(expected_interval, actual_interval)

  def test_raises_exception_if_alternative_is_not_expected(
      self,
  ):
    mean_1 = 2.5
    mean_2 = 2.0
    standard_error_1 = 1.0
    standard_error_2 = 0.5
    corr = 0.1
    degrees_of_freedom = 100

    with self.assertRaises(ValueError):
      statistics._absolute_difference_confidence_interval(
          mean_1=mean_1,
          mean_2=mean_2,
          standard_error_1=standard_error_1,
          standard_error_2=standard_error_2,
          corr=corr,
          degrees_of_freedom=degrees_of_freedom,
          alternative="something_wrong",
      )


class PowerTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="alternative=two_sided",
          changed_function_arguments=dict(alternative="two-sided"),
          expected_minimum_detectable_effect=5.657671912897953,
      ),
      dict(
          testcase_name="alternative=greater",
          changed_function_arguments=dict(alternative="greater"),
          expected_minimum_detectable_effect=5.006967533952627,
      ),
      dict(
          testcase_name="alternative=less",
          changed_function_arguments=dict(alternative="less"),
          expected_minimum_detectable_effect=5.006967533952627,
      ),
      dict(
          testcase_name="standard_error=3",
          changed_function_arguments=dict(standard_error=3.0),
          expected_minimum_detectable_effect=8.48650786934693,
      ),
      dict(
          testcase_name="degrees_of_freedom=200",
          changed_function_arguments=dict(degrees_of_freedom=200),
          expected_minimum_detectable_effect=5.630253045907562,
      ),
      dict(
          testcase_name="alpha=0.1",
          changed_function_arguments=dict(alpha=0.1),
          expected_minimum_detectable_effect=5.0068122095699445,
      ),
      dict(
          testcase_name="power=0.9",
          changed_function_arguments=dict(power=0.9),
          expected_minimum_detectable_effect=6.546216359712762,
      ),
  ])
  def test_calculate_minimum_detectable_effect_from_stats_returns_expected_effect(
      self, changed_function_arguments, expected_minimum_detectable_effect
  ):
    """All expected_minimum_detectable_effect values are from statsmodels.

    The expected results are calculated using
    statsmodels.stats.power.TTestPower. The method in statsmodels is numerical
    while ours is not, so we only test that the results are close. As long
    as the results are not different by more than 0.1% the difference is not
    of practical significance.
    """
    default_function_arguments = dict(
        standard_error=2.0,
        degrees_of_freedom=100,
        alternative="two-sided",
    )
    function_arguments = default_function_arguments | changed_function_arguments

    minimum_detectable_effect = (
        statistics.calculate_minimum_detectable_effect_from_stats(
            **function_arguments
        )
    )

    np.testing.assert_allclose(
        minimum_detectable_effect,
        expected_minimum_detectable_effect,
        rtol=0.001,
    )

  def test_calculate_minimum_detectable_effect_from_stats_raises_exception_for_unexpected_alternative(
      self,
  ):
    with self.assertRaises(ValueError):
      statistics.calculate_minimum_detectable_effect_from_stats(
          standard_error=2.0,
          degrees_of_freedom=100,
          alternative="bad_value",
      )

  @parameterized.parameters([
      ("two-sided", 1.13190873),
      ("greater", 1.00234224),
      ("less", 1.00234224),
  ])
  def test_yuens_t_test_ind_minimum_detectable_effect_returns_expected_result(
      self, alternative, expected_absolute_mde
  ):
    standard_deviation = 2.0
    sample_size = 100

    minimum_detectable_effect = (
        statistics.yuens_t_test_ind_minimum_detectable_effect(
            standard_deviation=standard_deviation,
            sample_size=sample_size,
            alternative=alternative,
        )
    )

    self.assertAlmostEqual(minimum_detectable_effect, expected_absolute_mde)

  @parameterized.parameters([
      ("two-sided", 0.52173106),
      ("greater", 0.46202054),
      ("less", 0.46202054),
  ])
  def test_yuens_t_test_paired_minimum_detectable_effect_returns_expected_result(
      self, alternative, expected_absolute_mde
  ):
    standard_deviation_1 = 2.0
    standard_deviation_2 = 3.0
    correlation = 0.8
    sample_size = 100

    difference_standard_deviation = np.sqrt(
        standard_deviation_1**2
        + standard_deviation_2**2
        - 2 * standard_deviation_1 * standard_deviation_2 * correlation
    )

    minimum_detectable_effect = (
        statistics.yuens_t_test_paired_minimum_detectable_effect(
            difference_standard_deviation=difference_standard_deviation,
            sample_size=sample_size,
            alternative=alternative,
        )
    )

    self.assertAlmostEqual(minimum_detectable_effect, expected_absolute_mde)


if __name__ == "__main__":
  absltest.main()
