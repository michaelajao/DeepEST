import plotly.graph_objects as go
import numpy as np


def plot_line_chart(actual_data, predicted_data, batch_names=None, location_names=None):
    # 检查数据形状是否相同且符合要求
    if actual_data.shape != predicted_data.shape:
        raise ValueError("The shapes of actual_data and predicted_data must be the same.")
    if actual_data.ndim not in [2, 3]:
        raise ValueError("Data should be either 2D (shape [B, F]) or 3D (shape [B, N, F])")

    # 设置默认的 batch_names 和 location_names
    if batch_names is None:
        batch_names = [f'Batch {i+1}' for i in range(actual_data.shape[0])]
    if location_names is None and actual_data.ndim == 3:
        location_names = [f'Location {j+1}' for j in range(actual_data.shape[1])]

    # 创建图形对象
    fig = go.Figure()

    # 添加轨迹
    num_traces = 0
    if actual_data.ndim == 2:
        # 二维数据 [B, F]
        for i in range(actual_data.shape[0]):
            x = np.arange(actual_data.shape[1])
            fig.add_trace(go.Scatter(x=x, y=actual_data[i], name=f'Actual {batch_names[i]}', visible=(i == 0)))
            fig.add_trace(go.Scatter(x=x, y=predicted_data[i], name=f'Predicted {batch_names[i]}', visible=(i == 0)))
            num_traces += 2
    else:
        # 三维数据 [B, N, F]
        for i in range(actual_data.shape[0]):
            for j in range(actual_data.shape[1]):
                x = np.arange(actual_data.shape[2])
                fig.add_trace(go.Scatter(x=x, y=actual_data[i, j], name=f'Actual {batch_names[i]} {location_names[j]}', visible=(i == 0 and j == 0)))
                fig.add_trace(go.Scatter(x=x, y=predicted_data[i, j], name=f'Predicted {batch_names[i]} {location_names[j]}', visible=(i == 0 and j == 0)))
                num_traces += 2

    # 创建下拉菜单
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

    # 更新图表布局
    fig.update_layout(
        title='Comparison of Actual and Predicted Data',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
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
                'x': 0.8,
                'xanchor': 'right',
                'y': 1,
                'yanchor': 'top'
            }
        ]
    )

    # 显示图形
    fig.show()


if __name__=="__main__":
    # 示例数据
    actual_data_2d = np.random.rand(2, 10)  # [B, F]
    predicted_data_2d = np.random.rand(2, 10)  # [B, F]
    batch_names_2d = ['Batch 1', 'Batch 2']

    actual_data_3d = np.random.rand(2, 3, 10)  # [B, N, F]
    predicted_data_3d = np.random.rand(2, 3, 10)  # [B, N, F]
    batch_names_3d = ['Batch A', 'Batch B']
    location_names_3d = ['Location 1', 'Location 2', 'Location 3']

    # 调用函数
    plot_line_chart(actual_data_2d, predicted_data_2d, batch_names=batch_names_2d)
    plot_line_chart(actual_data_3d, predicted_data_3d, batch_names=batch_names_3d, location_names=location_names_3d)
