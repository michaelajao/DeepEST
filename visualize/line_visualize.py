import plotly.graph_objects as go
import numpy as np


def plot_line_chart(actual_data, predicted_data, batch_names=None, location_names=None, x_axis_values: list= None):
    """
    Plots a line chart comparing actual and predicted data over time.

    Parameters:
    ----------
    actual_data : np.ndarray
        Array containing the actual data points. Should be 2D (shape [B, F]) or 3D (shape [B, N, F]).
    predicted_data : np.ndarray
        Array containing the predicted data points with the same shape as actual_data.
    batch_names : list of str, optional
        List of names for each batch in the data. If None, default names will be generated.
    location_names : list of str, optional
        List of names for each location in the data. If None and the data is 3D, default names will be generated.
    x_axis_values : list, optional
        List of values to be used on the x-axis. If None, indices will be used.

    Raises:
    ------
    ValueError
        If the shapes of actual_data and predicted_data do not match.
    ValueError
        If the ndim of data is not 2 or 3.
    AssertionError
        If the length of x_axis_values does not match the data dimensions.

    Returns:
    -------
    None
        Displays the line chart comparing actual and predicted data.
    """
    if actual_data.shape != predicted_data.shape:
        raise ValueError("The shapes of actual_data and predicted_data must be the same.")
    if actual_data.ndim not in [2, 3]:
        raise ValueError("Data should be either 2D (shape [B, F]) or 3D (shape [B, N, F])")

    if batch_names is None:
        batch_names = [f'Batch {i+1}' for i in range(actual_data.shape[0])]
    if location_names is None and actual_data.ndim == 3:
        location_names = [f'Location {j+1}' for j in range(actual_data.shape[1])]

    fig = go.Figure()
    num_traces = 0
    if actual_data.ndim == 2:
        for i in range(actual_data.shape[0]):
            if x_axis_values is not None:
                if len(x_axis_values) != actual_data.shape[1]:
                    raise AssertionError("x_axis_values's lenght don't match the data")
                else:
                    x = x_axis_values
            else:        
                x = np.arange(actual_data.shape[1])
            fig.add_trace(go.Scatter(x=x, y=actual_data[i], name=f'Actual {batch_names[i]}', visible=(i == 0)))
            fig.add_trace(go.Scatter(x=x, y=predicted_data[i], name=f'Predicted {batch_names[i]}', visible=(i == 0)))
            num_traces += 2
    else:
        for i in range(actual_data.shape[0]):
            for j in range(actual_data.shape[1]):
                if x_axis_values is not None:
                    if len(x_axis_values) != actual_data.shape[2]:
                        raise AssertionError("x_axis_values's lenght don't match the data")
                    else:
                        x = x_axis_values
                else:
                    x = np.arange(actual_data.shape[2])
                fig.add_trace(go.Scatter(x=x, y=actual_data[i, j], name=f'Actual {batch_names[i]} {location_names[j]}', visible=(i == 0 and j == 0)))
                fig.add_trace(go.Scatter(x=x, y=predicted_data[i, j], name=f'Predicted {batch_names[i]} {location_names[j]}', visible=(i == 0 and j == 0)))
                num_traces += 2

    buttons_batch = []
    buttons_location = []

    if actual_data.ndim == 2:
        for i in range(actual_data.shape[0]):
            buttons_batch.append({
                'label': batch_names[i],
                'method': 'update',
                'args': [{'visible': [False] * num_traces}]
            })
            buttons_batch[-1]['args'][0]['visible'][i*2] = True
            buttons_batch[-1]['args'][0]['visible'][i*2 + 1] = True
    else:
        for i in range(actual_data.shape[0]):
            buttons_batch.append({
                'label': batch_names[i],
                'method': 'update',
                'args': [{'visible': [False] * num_traces}]
            })
            for j in range(actual_data.shape[1]):
                buttons_batch[-1]['args'][0]['visible'][(i * actual_data.shape[1] + j) * 2] = (j == 0)
                buttons_batch[-1]['args'][0]['visible'][(i * actual_data.shape[1] + j) * 2 + 1] = (j == 0)

        for j in range(actual_data.shape[1]):
            buttons_location.append({
                'label': location_names[j],
                'method': 'update',
                'args': [{'visible': [False] * num_traces}]
            })
            for i in range(actual_data.shape[0]):
                buttons_location[-1]['args'][0]['visible'][(i * actual_data.shape[1] + j) * 2] = (i == 0)
                buttons_location[-1]['args'][0]['visible'][(i * actual_data.shape[1] + j) * 2 + 1] = (i == 0)

    fig.update_layout(
        title='Comparison of Actual and Predicted Data',
        xaxis_title='Date',
        xaxis_title_font=dict(size=16),
        yaxis_title='Infected cases',
        yaxis_title_font=dict(size=16),
        xaxis=dict(
            tickformat='%Y-%m-%d',  # 设置 x 轴日期格式,
        ),
        updatemenus=[
            {
                'type': 'dropdown',
                'direction': 'down',
                'buttons': buttons_batch,
                'showactive': True,
                'x': 1,
                'xanchor': 'right',
                'y': 1,
                'yanchor': 'top'
            },
            {
                'type': 'dropdown',
                'direction': 'down',
                'buttons': buttons_location,
                'showactive': True,
                'x': 0.85,
                'xanchor': 'right',
                'y': 1,
                'yanchor': 'top'
            }
        ],
        font=dict(size=16),  # 设置图例和文本的默认字体大小
        legend=dict(font=dict(size=16)) 
    )

    fig.show()


if __name__=="__main__":
    actual_data_2d = np.random.rand(2, 10)  # [B, F]
    predicted_data_2d = np.random.rand(2, 10)  # [B, F]
    batch_names_2d = ['Batch 1', 'Batch 2']

    actual_data_3d = np.random.rand(2, 3, 10)  # [B, N, F]
    predicted_data_3d = np.random.rand(2, 3, 10)  # [B, N, F]
    batch_names_3d = ['Batch A', 'Batch B']
    location_names_3d = ['Location 1', 'Location 2', 'Location 3']

    plot_line_chart(actual_data_2d, predicted_data_2d, batch_names=batch_names_2d)
    plot_line_chart(actual_data_3d, predicted_data_3d, batch_names=batch_names_3d, location_names=location_names_3d)
