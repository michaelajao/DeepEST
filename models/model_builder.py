from dl_models import cola_gnn, GRU, Transformer, RNN, LSTM, STPModel,STAN
from ml_models import ARIMA, SIR, SEIR, XGBoost
from typing import Optional
import torch
import dgl
import sys
sys.path.append("../")
from datasets import PreprocessData

dl_model_dict = {
    "colagnn": cola_gnn,
    "gru": GRU,
    "stp": STPModel,
    "transformer": Transformer,
    "rnn": RNN,
    "lstm": LSTM,
    "stan": STAN
}

ml_model_dict = {
    "arima": ARIMA,
    "sir": SIR,
    "seir": SEIR,
    "xgboost": XGBoost
}


def build_model(
    preprocess: PreprocessData,
    model_name: str,
    hidden_size: int = 64,
    edge_index: Optional[torch.sparse_coo_tensor] = None,  # torch.sparse_coo_tensor()
    **kwargs,
):
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
        input_size = preprocess.getFeaturesNum()
        output_size = preprocess.output_window
        print(input_size, output_size)
        model_instance = model_class(input_size, hidden_size, output_size)
    elif model_name == "stp":
        input_size = preprocess.getFeaturesNum()
        output_size = preprocess.output_window
        edge_index = edge_index.coalesce().indices()
        model_instance = model_class(input_size=input_size,
                                     hidden_size=hidden_size,
                                     output_size=output_size,
                                     edge_index=edge_index,
                                     )
    elif model_name == "colagnn":
        input_size = preprocess.getFeaturesNum()
        num_positions = preprocess.getPositionNum()
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
        input_size = preprocess.getFeaturesNum()
        output_window = preprocess.output_window
        rows, cols = edge_index.coalesce().indices()
        g = dgl.DGLGraph()
        g.add_nodes(edge_index.size(0))
        g.add_edges(rows, cols)
        model_instance = model_class(g, input_size, output_window)
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
