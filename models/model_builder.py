from dl_models import cola_gnn, GRU, Transformer, RNN, LSTM, STPModel,STAN, HOIST, TAPRSV
from ml_models import ARIMA, SIR, SEIR, XGBoost, RandomForest, LightGBM, SVM
from typing import Optional
import torch
import dgl
import sys
sys.path.append("../")
from datasets import preprocess_data

dl_model_dict = {
    "colagnn": cola_gnn,
    "gru": GRU,
    "stp": STPModel,
    "transformer": Transformer,
    "rnn": RNN,
    "lstm": LSTM,
    "stan": STAN,
    "hoist": HOIST,
    "taprsv": TAPRSV
}

ml_model_dict = {
    "arima": ARIMA,
    "sir": SIR,
    "seir": SEIR,
    "xgboost": XGBoost,
    'lightgbm': LightGBM,
    'random_forest': RandomForest,
    'svm': SVM
}


def build_model(
    preprocess: preprocess_data,
    model_name: str,
    hidden_size: int = 64,
    edge_index: Optional[torch.sparse_coo_tensor] = None,  # torch.sparse_coo_tensor()
    **kwargs,
):
    """
    Build and return an instance of a specified model for deep learning or machine learning.

    Parameters:
    ----------
    preprocess : preprocess_data
        Instance of preprocessed data containing necessary metadata for model construction.
    model_name : str
        Name of the model to instantiate.
    hidden_size : int, optional
        Size of the hidden layers in the model, by default 64.
    edge_index : torch.sparse_coo_tensor, optional
        The edge index tensor for graph-based models, by default None.
    **kwargs : dict
        Additional keyword arguments passed to the model constructor.

    Returns:
    -------
    model_instance
        An instance of the specified model.

    Raises:
    ------
    AssertionError
        If the model name does not exist in the dictionaries.
    """
    if kwargs:
        modified_kwargs = {key.upper(): value for key, value in kwargs.items()}
    model_name = model_name.lower()
    if model_name in dl_model_dict:
        model_class = dl_model_dict[model_name]
    elif model_name in ml_model_dict:
        model_class = ml_model_dict[model_name]
    else:
        raise AssertionError("No such name in ml_models or dl_models")
    if model_name == "gru" or model_name == "rnn" or model_name == "lstm":
        input_size = preprocess.get_features_num()
        output_size = preprocess.output_window
        print(input_size, output_size)
        model_instance = model_class(input_size, hidden_size, output_size)
    elif model_name == "stp":
        input_size = preprocess.get_features_num()
        output_size = preprocess.output_window
        edge_index = edge_index.coalesce().indices()
        model_instance = model_class(input_size=input_size,
                                     hidden_size=hidden_size,
                                     output_size=output_size,
                                     edge_index=edge_index,
                                     )
    elif model_name == "colagnn":
        input_size = preprocess.get_features_num()
        num_positions = preprocess.get_position_num()
        input_window = preprocess.input_window
        output_window = preprocess.output_window
        adj = edge_index.to_dense()
        model_instance = model_class(input_size,
                                     num_positions,
                                     input_window,
                                     output_window,
                                     adj,
                                     hidden_size
                                     )
    elif model_name == "stan":
        input_size = preprocess.get_features_num()
        output_window = preprocess.output_window
        rows, cols = edge_index.coalesce().indices()
        g = dgl.DGLGraph()
        g.add_nodes(edge_index.size(0))
        g.add_edges(rows, cols)
        model_instance = model_class(g, input_size, output_window)
    elif model_name == "hoist":
        dynamic_dims = preprocess.get_dynamic_dim()
        static_dims = preprocess.get_static_dim()
        total_dims = preprocess.get_features_num()
        input_window = preprocess.input_window
        output_window = preprocess.output_window
        if static_dims != 0:
            model_instance = model_class(dynamic_dims= [dynamic_dims], static_dims= [static_dims], input_window= input_window, output_window= output_window)
        else:
            model_instance = model_class(dynamic_dims= [total_dims], input_window= input_window, output_window= output_window)
    elif model_name == 'taprsv':
        dynamic_dims = preprocess.get_dynamic_dim()
        static_dims = preprocess.get_static_dim()
        total_dims = preprocess.get_features_num()
        input_window = preprocess.input_window
        output_window = preprocess.output_window
        num_positions = preprocess.get_position_num()
        model_instance = model_class(num_positions = num_positions, dynamic_dims= dynamic_dims, static_dims= static_dims, output_window= output_window)
    else:
        model_class = ml_model_dict[model_name]
        if model_class is SIR:
            model_instance = model_class(S0=modified_kwargs["S0"],
                                         I0=modified_kwargs["I0"],
                                         R0=modified_kwargs["R0"])
        elif model_class is SEIR:
            model_instance = model_class(S0=modified_kwargs["S0"],
                                         E0=modified_kwargs["E0"],
                                         I0=modified_kwargs["I0"],
                                         R0=modified_kwargs["R0"])
        else:
            model_instance = model_class()
        return model_instance
    return model_instance
