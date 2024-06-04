import torch
from sklearn.metrics import *
from scipy.stats import pearsonr
import numpy as np


def getMetrics(y_forecast: torch.Tensor, y_target: torch.Tensor):
    """Computes metrics.

    User can specify which metrics to compute by passing a list of metric names.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_target: Tensor.
        y_forecast: Tensor.
        metrics: List of metrics to compute. Default is ['MSE', 'RMSE', 'MAE', 'R2_SCORE', 'MAPE', 'PEARSON_CORRELATION'].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import binary_metrics_fn
        >>> y_target = torch.Tensor([0, 0, 1, 1])
        >>> y_forecast = torch.Tensor([0.1, 0.4, 0.35, 0.8])
        >>>
    """

    metrics = ['MSE', 'RMSE', 'MAE', 'R2_SCORE', 'MAPE','PEARSON_CORRELATION']
    output = {}
    if (isinstance(y_forecast, torch.Tensor)):
        y_forecast = y_forecast.cpu().tolist()
    if (isinstance(y_target, torch.Tensor)):
        y_target = y_target.cpu().tolist()
    if (isinstance(y_forecast, np.ndarray)):
        y_forecast = y_forecast.tolist()
    if (isinstance(y_target, np.ndarray)):
        y_target = y_target.tolist()

    for metric in metrics:
        if metric == 'MSE':
            output[metric] = mean_squared_error(y_forecast, y_target)
        elif metric == 'RMSE':
            output[metric] = np.sqrt(mean_squared_error(y_forecast, y_target))
        elif metric == 'MAE':
            output[metric] = mean_absolute_error(y_forecast, y_target)
        elif metric == 'R2_SCORE':
            output[metric] = r2_score(y_forecast, y_target)
        elif metric == 'MAPE':
            output[metric] = mean_absolute_percentage_error(y_forecast, y_target)
        elif metric == 'PEARSON_CORRELATION':
            corr, _ = pearsonr(y_forecast, y_target)
            output[metric] = corr
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return output


if __name__ == "__main__":
    y_target = torch.randn(12)
    y_forecast = torch.randn(12)
    print(getMetrics(y_forecast=y_forecast, y_target=y_target))
