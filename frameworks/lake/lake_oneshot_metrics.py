"""lake_oneshot_metrics.py"""

import numpy as np

from utils import compute_matching

class LakeOneshotMetrics:
  """
  A self-contained module to compute, manage and report the metrics for one-shot learning.

  The metrics are based on similarity/matching metrics used in the Lake's Omniglot experiments.
  """

  match_mse_key = 'match_mse'
  match_acc_mse_key = 'acc_mse'
  sum_ambiguous_mse_key = 'amb_mse'

  match_mse_tf_key = 'match_mse_tf'
  match_acc_mse_tf_key = 'acc_mse_tf'
  sum_ambiguous_mse_tf_key = 'amb_mse_tf'

  match_olap_key = 'matching_matrices_olp'
  match_olap_tf_key = 'matching_matrices_olp_tf'
  match_acc_olap_key = 'acc_olp'
  match_acc_olap_tf_key = 'acc_olp_tf'
  sum_ambiguous_olap_key = 'amb_olp'
  sum_ambiguous_olap_tf_key = 'amb_olp_tf'

  def __init__(self):
    self.average_metrics = {}

    self.matching_matrix_keys = [
        self.match_mse_key,
        self.match_mse_tf_key,
        self.match_olap_key,
        self.match_olap_tf_key
    ]

    self.matching_accuracies_keys = [
        self.match_acc_mse_key,
        self.match_acc_mse_tf_key,
        self.match_acc_olap_key,
        self.match_acc_olap_tf_key
    ]

    self.sum_ambiguous_keys = [
        self.sum_ambiguous_mse_key,
        self.sum_ambiguous_mse_tf_key,
        self.sum_ambiguous_olap_key,
        self.sum_ambiguous_olap_tf_key
    ]

  def get_keys(self):
    return (
        self.matching_matrix_keys,
        self.matching_accuracies_keys,
        self.sum_ambiguous_keys
    )

  def update_averages(self, key, value):
    if key not in self.average_metrics.keys():
      self.average_metrics[key] = []

    self.average_metrics[key].append(value)

  def compute(self, study_features, recall_features, modes):
    """Compute similarity matrix and accuracy for all metrics."""
    metrics = {}
    metric_keys = zip(*self.get_keys())

    comparison_types = ['mse', 'mse', 'overlap', 'overlap']
    train_features = [study_features, recall_features, study_features, recall_features]
    test_features = [recall_features, study_features, recall_features, study_features]

    for i, (match_key, match_acc_key, sub_amb_key) in enumerate(metric_keys):
      matching_matrices, matching_accuracies, sum_ambiguous = compute_matching(
          modes, train_features[i], test_features[i], comparison_types[i])

      metrics[match_key] = matching_matrices
      metrics[match_acc_key] = matching_accuracies
      metrics[sub_amb_key] = sum_ambiguous

    return metrics

  def report(self, results, verbose=True):
    """Format and report specified metrics, and update running averages."""
    matching_matrix_keys, matching_accuracies_keys, sum_ambiguous_keys = self.get_keys()
    skip_console = matching_matrix_keys + matching_accuracies_keys + sum_ambiguous_keys

    def log_to_console(x):
      if not verbose:
        return
      print(x)

    log_to_console("\n--------- General Metrics -----------")
    np.set_printoptions(threshold=np.inf)

    for metric, metric_value in results.items():
      if metric not in skip_console:
        log_to_console("\t{0} : {1}".format(metric, metric_value))

        self.update_averages('{}'.format(metric), metric_value)

    log_to_console("\n--------- Oneshot/Lake Metrics (i.e. PC fed back through AMTL) -----------")

    for matching_accuracies_key in matching_accuracies_keys:    # for different comparison types
      matching_accuracies = results[matching_accuracies_key]

      for accuracy_type, val in matching_accuracies.items():    # for different features
        log_to_console("\t{}_{} : {:.3f}".format(matching_accuracies_key, accuracy_type, val))

        self.update_averages('{}_{}'.format(matching_accuracies_key, accuracy_type), val)

    for sum_ambiguous_key in sum_ambiguous_keys:   # for different comparison types
      sum_ambiguous = results[sum_ambiguous_key]

      for sum_ambiguous_type, val in sum_ambiguous.items():   # for different feature types
        log_to_console("\t{}_{} : {:.3f}".format(sum_ambiguous_key, sum_ambiguous_type, val))

        self.update_averages('{}_{}'.format(sum_ambiguous_key, sum_ambiguous_type), val)

  def report_averages(self, export_csv=True):
    """Report the averaged metrics, and optionally export a CSV-friendly format."""
    if not self.average_metrics:
      return

    print("\n--------- Averages for all batches: ----------")
    for accuracy_type, vals in self.average_metrics.items():
      av = np.mean(vals, dtype=np.float64)
      print("\t{}: {}     (length={})".format(accuracy_type, av, len(vals)))

    print('\n')

    if export_csv:
      # print as comma separated list for import into a spreadsheet
      # headings
      for accuracy_type, vals in self.average_metrics.items():
        print("{}, ".format(accuracy_type), end='')
      print('\n')

      # values
      for accuracy_type, vals in self.average_metrics.items():
        av = np.mean(vals, dtype=np.float64)
        print("{}, ".format(av), end='')
      print('\n')
