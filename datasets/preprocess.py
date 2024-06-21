import torch
from typing import Optional
from .dataspliter import split_by_spatial_temporal
from .dataset import dl_dataset, ml_dataset


def merge_timestep(
    dynamic_data,
    label,
    merge_len: int = 1,
    mode: str = "sum",
) -> torch.Tensor:
    """
    Merges timesteps in dynamic data and label tensors based on the specified mode.

    Parameters:
    ----------
    dynamic_data : torch.Tensor
        Tensor containing dynamic data with shape (N, T, D).
    label : torch.Tensor
        Tensor containing labels with shape (N, T, D).
    merge_len : int, optional
        Number of timesteps to merge, by default 1.
    mode : str, optional
        The method of merging, either 'sum' or 'mean', by default 'sum'.

    Returns:
    -------
    torch.Tensor, torch.Tensor
        Merged dynamic data and label tensors.

    Raises:
    ------
    AssertionError
        If an unknown merge mode is provided.

    Examples:
    --------
    >>> dynamic_data = torch.randn(10, 5, 3)  # Example dynamic data
    >>> label = torch.randn(10, 5, 3)        # Corresponding labels
    >>> merged_data, merged_label = merge_timestep(dynamic_data, label, merge_len=2, mode='mean')
    >>> print(merged_data.shape, merged_label.shape)
    # Output: (torch.Size([10, 3, 3]), torch.Size([10, 3, 3]))
    """
    dynamic_data = torch.Tensor(dynamic_data)
    label = torch.Tensor(label)
    _, num_time, _ = dynamic_data.shape
    new_data = torch.Tensor([])
    new_label = torch.Tensor([])
    cur_step = 0
    # print(f'self.dynamic_data.shape is {self.dynamic_data.shape}')
    while cur_step + merge_len <= num_time:
        # print(f'cur_step is {cur_step}')
        tmp_data = dynamic_data[:, cur_step: cur_step + merge_len, :]
        tmp_label = label[:, cur_step: cur_step + merge_len, :]
        # print(f'tmp_data.shape is {tmp_data.shape}')
        if mode == 'sum':
            data_item = torch.sum(tmp_data, dim=1, keepdim=True)
            label_item = torch.sum(tmp_label, dim=1, keepdim=True)
            # print(f'data_itme.shape is {data_item.shape}')
        elif mode == 'mean':
            data_item = torch.mean(tmp_data, dim=1, keepdim=True)
            label_item = torch.mean(tmp_label, dim=1, keepdim=True)
            # print(f'data_itme.shape is {data_item.shape}')
        # todo flatten 如何处理label?
        # elif mode == 'flatten':
        #     data_item = torch.flatten(tmp_data, start_dim= 1, end_dim= 2)
        #     data_item = torch.unsqueeze(data_item, 1)
        #     print(f'data_itme.shape is {data_item.shape}')
        else:
            raise AssertionError("No such merge mode! We only provide 'sum' or\
                                 'mean'.")
        new_data = torch.cat((new_data, data_item), 1)
        new_label = torch.cat((new_label, label_item), 1)
        # print(f'new_data.shape is {new_data.shape}')
        cur_step = cur_step + merge_len
    return new_data, new_label


class temporal_data_preprocess():
    
    """
    Preprocessing class for temporal data models.

    Attributes:
    -----------
    dynamic_data : torch.Tensor
        The dynamic data tensor to preprocess.
    static_data : torch.Tensor
        The static data tensor, if used.
    label : torch.Tensor
        The label tensor after preprocessing.
    input_window : int
        The size of the input window for the model.
    output_window : int
        The size of the output window for the model.
    data_mixed : bool
        Whether dynamic and static data are mixed.
    normalization : str
        The normalization method applied, if any.
    stride : int
        The stride used when creating timesteps.
    kwargs: define the mode of how to choose geometirc information for
    temporal-only models, the format of kwargs is
    {
        mode:0,
        method:'sum' or 'mean'
    }, which means in mode 0, we will use the features of all regions.
    The specific way to use is: sum or average the features
    {
        mode:1
        place:0 -> N
    }, which means in mode 1, we only use the features of a certain place,
    which is specified by the place parameter,
    which ranges from 0 to N-1. Where N is the total number of places

    Examples:
    --------
    >>> preprocessor = temporal_data_preprocess(dynamic_data, static_data, label)
    >>> processed_data, processed_label = preprocessor.get_data(), preprocessor.get_label()
    >>> print(processed_data.shape, processed_label.shape)
    # Output: (torch.Size([batches, timesteps, features]), torch.Size([batches, timesteps, 1]))
    """
    def __init__(
        self,
        dynamic_data: torch.Tensor,
        static_data: torch.Tensor,
        label: torch.Tensor,
        input_window: int = 1,
        output_window: int = 1,
        data_mixed: bool = True,
        normalization: str = 'z-score',
        stride: int = 1,
        **kwargs
    ):
        self.dynamic_data = dynamic_data
        self.static_data = static_data
        self.label = label
        self.input_window = input_window
        self.output_window = output_window
        self.kwargs = kwargs
        self.normalization = normalization
        self.stride = stride
        self.__select_mode__()
        if data_mixed is False:
            self.origin_data = self.dynamic_data
            self.origin_label = self.label
        else:
            self.origin_data = self.__ConcatData__()
            self.origin_label = self.label

        if normalization is not None:
            self.norm_data, self.norm_label = self.__normalization__()

        else:
            self.norm_data = self.origin_data
            self.norm_label = self.origin_label

        self.data, self.label = self.__SetTimestep__()

    def __select_mode__(self):
        '''
        after this function, the shape of dynamic_data will
        change from (N, T, D) to (T, D),
        the shape of static_data will change from (N, S) to (S)
        '''
        if self.kwargs['mode'] == 0:
            if self.kwargs['method'] == 'sum':
                self.dynamic_data = torch.sum(self.dynamic_data, 0)
                self.static_data = torch.sum(self.static_data, 0)
                self.label = torch.sum(self.label, 0)
            elif self.kwargs['method'] == 'mean':
                self.dynamic_data = torch.mean(self.dynamic_data, 0)
                self.static_data = torch.mean(self.static_data, 0)
                self.label = torch.mean(self.label, 0)
            else:
                raise AssertionError("No such mode! We only provide sum or mean.")
        elif self.kwargs['mode'] == 1:
            num_positions = self.dynamic_data.shape[0]
            if not isinstance(self.kwargs['place'], int):
                raise AssertionError("Input Error, please set the place \
                                     parameter as int")
            elif self.kwargs['place'] < num_positions and self.kwargs['place'] >= 0:
                self.dynamic_data = self.dynamic_data[self.kwargs['place'], :, :]
                self.static_data = self.static_data[self.kwargs['place'], :]
                self.label = self.label[self.kwargs['place'], :, :]
            else:
                raise AssertionError("Index Error, please select place from 0 to N - 1.")
        else:
            raise AssertionError("No such mode! please set mode as 0 or 1!")

    def __ConcatData__(self):
        ex_dim = self.dynamic_data.shape[0]
        self.static_data = torch.unsqueeze(self.static_data, dim=0)
        self.static_data = self.static_data.repeat(ex_dim, 1)
        return torch.cat((self.static_data, self.dynamic_data), dim=-1)

    def __normalization__(self):
        num_features = self.origin_data.shape[-1]
        self.data_min = torch.empty((1, num_features))
        self.data_max = torch.empty((1, num_features))
        self.data_mean = torch.empty((1, num_features))
        self.data_std = torch.empty((1, num_features))
        for i in range(num_features):
            self.data_min[0, i] = self.origin_data[:, i].min()
            self.data_max[0, i] = self.origin_data[:, i].max()
            self.data_std[0, i] = self.origin_data[:, i].std()
            self.data_mean[0, i] = self.origin_data[:, i].mean()
        self.label_min = torch.min(self.origin_label)
        self.label_max = torch.max(self.origin_label)
        self.label_mean = torch.mean(self.origin_label)
        self.label_std = torch.std(self.origin_label)
        if self.normalization == 'z-score':
            data_norm = (self.origin_data - self.data_mean) / (self.data_std + 1e-6)
            label_norm = (self.origin_label - self.label_mean) / (self.label_std + 1e-6)
        elif self.normalization == 'min-max':
            data_norm = (self.origin_data - self.data_min) / (self.data_max - self.data_min + 1e-6)
            label_norm = (self.origin_label - self.label_min) / (self.label_max - self.label_min + 1e-6)
        else:
            raise AssertionError("No such normalization method! We only provide 'min-max' or 'z-score'.")
        return data_norm, label_norm

    def __SetTimestep__(self):
        timestep_num = self.norm_data.shape[0]
        data_start = 0
        data_end = self.input_window
        label_start = self.input_window
        label_end = self.input_window + self.output_window
        input_data = torch.Tensor()
        input_label = torch.Tensor()
        while label_end <= timestep_num:
            temp_data = self.norm_data[data_start:data_end, :]
            temp_label = self.norm_label[label_start:label_end]
            if data_start == 0:
                input_data = torch.unsqueeze(temp_data.clone(), 0)
                input_label = torch.unsqueeze(temp_label.clone(), 0)
            else:
                input_data = torch.cat((input_data, torch.unsqueeze(temp_data, 0)), 0)
                input_label = torch.cat((input_label, torch.unsqueeze(temp_label, 0)), 0)

            label_start += self.stride
            label_end += self.stride
            data_start += self.stride
            data_end += self.stride
        # print(f'input_data.shape is {input_data.shape}, input_label.shape is {input_label.shape}')
        return input_data, input_label

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def get_ml_data(self):
        return self.origin_data

    def get_ml_label(self):
        return self.origin_label
    
    def get_origin_data(self):
        return self.origin_data

    def get_origin_label(self):
        return self.origin_label

    def reverse_normalization(self, label_norm, data_norm=None):
        if self.normalization == 'z-score':
            if data_norm is not None:
                data = data_norm * (self.data_std + 1e-6) + self.data_mean
            label = label_norm * (self.label_std + 1e-6) + self.label_mean
        elif self.normalization == 'min-max':
            if data_norm is not None:
                data = data_norm * (self.data_max - self.data_min + 1e-6) + self.data_min
            label = label_norm * (self.label_max - self.label_min + 1e-6) + self.label_min
        else:
            raise AssertionError("No such normalization method! We only provide 'min-max' or 'z-score'.")
        if data_norm is not None:
            return data, label
        return label


class spatial_temporal_data_preprocess():
    """
    Preprocessing class for spatial-temporal data models.

    Attributes:
    -----------
    dynamic_data : torch.Tensor
        The dynamic data tensor to preprocess.
    static_data : torch.Tensor
        The static data tensor to preprocess.
    label : torch.Tensor
        The label tensor to preprocess.
    input_window : int
        The size of the input window for the model.
    output_window : int
        The size of the output window for the model.
    normalization : str
        The normalization method applied, if any.
    stride : int
        The stride used when creating timesteps.

    Examples:
    --------
    >>> preprocessor = spatial_temporal_data_preprocess(dynamic_data, static_data, input_window, output_window, label)
    >>> processed_data, processed_label = preprocessor.get_data(), preprocessor.get_label()
    >>> print(processed_data.shape, processed_label.shape)
    # Output: (torch.Size([batches, locations, input_window, features]), torch.Size([batches, locations,  output_window, 1]))
    """
    def __init__(
        self,
        dynamic_data: torch.Tensor,
        static_data: torch.Tensor,
        label: torch.Tensor,
        input_window: int = 1,
        output_window: int = 1,
        data_mixed: bool = True,
        normalization: str = 'z-score',
        stride: int = 1
    ):
        self.dynamic_data = torch.Tensor(dynamic_data)
        self.static_data = torch.Tensor(static_data)
        self.label = torch.Tensor(label)
        self.input_window = input_window
        self.output_window = output_window
        self.normalization = normalization
        self.stride = stride
        '''
        print(f'self.dynamic_data.shape is {self.dynamic_data.shape},
        self.static_data.shape is {self.static_data.shape}')
        '''
        if data_mixed is False:
            self.origin_data = self.dynamic_data
            self.origin_label = self.label
        else:
            self.origin_data = self.__ConcatData__()
            self.origin_label = self.label
        # print(f'self.origin_data.shape is {self.origin_data.shape}')

        if normalization is not None:
            self.norm_data, self.norm_label = self.__normalization__()

        else:
            self.norm_data = self.origin_data
            self.norm_label = self.origin_label

        self.data, self.label = self.__SetTimestep__()
        '''
        print(f'self.data.shape is {self.data.shape}
        self.label.shape is {self.label.shape}')
        '''

    # Concatenate dynamic and static features
    # (N,T,D),(N,S)
    def __ConcatData__(self):
        ex_dim = self.dynamic_data.shape[1]
        # print(f'static_data.shape0 is {self.static_data.shape}')
        self.static_data = torch.unsqueeze(self.static_data, dim=1)
        # print(f'static_data.shape1 is {self.static_data.shape}')
        self.static_data = self.static_data.repeat(1, ex_dim, 1)
        # print(f'static_data.shape2 is {self.static_data.shape}')
        return torch.cat((self.static_data, self.dynamic_data), dim=-1)

    # The features were normalized by z-score and min-max by this function
    def __normalization__(self):
        num_features = self.origin_data.shape[-1]
        self.data_min = torch.empty((1, 1, num_features))
        self.data_max = torch.empty((1, 1, num_features))
        self.data_mean = torch.empty((1, 1, num_features))
        self.data_std = torch.empty((1, 1, num_features))
        for i in range(num_features):
            self.data_min[0, 0, i] = self.origin_data[:, :, i].min()
            self.data_max[0, 0, i] = self.origin_data[:, :, i].max()
            self.data_std[0, 0, i] = self.origin_data[:, :, i].std()
            self.data_mean[0, 0, i] = self.origin_data[:, :, i].mean()
        self.label_min = torch.min(self.origin_label)
        self.label_max = torch.max(self.origin_label)
        self.label_mean = torch.mean(self.origin_label)
        self.label_std = torch.std(self.origin_label)
        # print('self.origin_data.shape is {}, data_min.shape is {}'.format(self.origin_data.shape, data_min.shape))
        if self.normalization == 'z-score':
            data_norm = (self.origin_data - self.data_mean) / (self.data_std + 1e-6)
            label_norm = (self.origin_label - self.label_mean) / (self.label_std + 1e-6)
        elif self.normalization == 'min-max':
            data_norm = (self.origin_data - self.data_min) / (self.data_max - self.data_min + 1e-6)
            label_norm = (self.origin_label - self.label_min) / (self.label_max - self.label_min + 1e-6)
        else:
            raise AssertionError("No such normalization method! We only provide 'min-max' or 'z-score'.")
        return data_norm, label_norm

    # 输入原始数据(N,T,D)和原始标签(N,T,1),输出按照input_window和output_window划分的M组数据(M, N, input_window, D)和M组标签(M, N, output_windwo, 1)
    def __SetTimestep__(self):
        timestep_num = self.norm_data.shape[1]
        data_start = 0
        data_end = self.input_window
        label_start = self.input_window
        label_end = self.input_window + self.output_window
        input_data = torch.Tensor()
        input_label = torch.Tensor()
        while label_end <= timestep_num:
            temp_data = self.norm_data[:, data_start:data_end, :]
            temp_label = self.norm_label[:, label_start:label_end, :]
            if data_start == 0:
                input_data = torch.unsqueeze(temp_data.clone(), 0)
                input_label = torch.unsqueeze(temp_label.clone(), 0)
            else:
                input_data = torch.cat((input_data, torch.unsqueeze(temp_data, 0)), 0)
                input_label = torch.cat((input_label, torch.unsqueeze(temp_label, 0)), 0)
            label_start += self.stride
            label_end += self.stride
            data_start += self.stride
            data_end += self.stride

        # print(f'input_data.shape is {input_data.shape}, input_label.shape is {input_label.shape}')
        return input_data, input_label

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def get_origin_data(self):
        return self.origin_data

    def get_origin_label(self):
        return self.origin_label

    def reverse_normalization(self,  label_norm, data_norm= None):
        if self.normalization == 'z-score':
            if data_norm is not None:
                data = data_norm * (self.data_std + 1e-6) + self.data_mean
            label = label_norm * (self.label_std + 1e-6) + self.label_mean
        elif self.normalization == 'min-max':
            if data_norm is not None:
                data = data_norm * (self.data_max - self.data_min + 1e-6) + self.data_min
            label = label_norm * (self.label_max - self.label_min + 1e-6) + self.label_min
        else:
            raise AssertionError("No such normalization method! We only provide 'min-max' or 'z-score'.")
        if data_norm is not None:
            return data, label
        return label

class preprocess_data():
    """
    Class for preprocessing data for machine learning models, supporting both
    temporal and spatial-temporal data types.

    Parameters:
    ----------
    dynamic_data : torch.Tensor
        The dynamic data tensor to preprocess.
    static_data : torch.Tensor
        The static data tensor to preproces.
    label : torch.Tensor
        The label tensor to preprocess.
    input_window : int, optional
        The size of the input window, by default 1.
    output_window : int, optional
        The size of the output window, by default 1.
    data_mixed : bool, optional
        Whether to mix dynamic and static data, by default True.
    normalization : str, optional
        The normalization method to use ('z-score' or 'min-max'), by default 'z-score'.
    stride : int, optional
        The stride for creating timesteps, by default 1.
    type : str, optional
        The type of data ('temporal' or 'spatial-temporal'), by default 'temporal'.
    temporal_rate : list, optional
        The rate for temporal splitting, by default [0.6, 0.2, 0.2].
    spatial_indexes : list, optional
        The indexes for spatial-temporal splitting, by default None.
    **kwargs : dict, optional
        Additional keyword arguments for preprocessing modes.

    Attributes:
    -----------
    train_process, val_process, test_process :
        The preprocessing objects for training, validation, and testing data sets.

    Examples:
    --------
    >>> preprocessor = preprocess_data(dynamic_data, static_data, labels, type='spatial-temporal')
    >>> train_data, train_label, val_data, val_label, test_data, test_label = preprocessor.get_data()
    >>> print(train_data.shape, train_label.shape)
    # Output: (torch.Size([batches, locations, input_window, features]), torch.Size([batches, locations,  output_window, 1]))
    """
    def __init__(
        self,
        dynamic_data: torch.Tensor,
        static_data: torch.Tensor,
        label: torch.Tensor,
        input_window: int = 1,
        output_window: int = 1,
        data_mixed: bool = True,
        normalization: str = 'z-score',
        stride: int = 1,
        type: str = 'temporal' or 'spatial-temporal',
        temporal_rate: list = [0.6, 0.2, 0.2],
        spatial_indexes: list = None,
        **kwargs: Optional[dict],
    ):
        """
        Constructs the preprocess_data object and initializes preprocessing pipelines.
        """
        if type != 'temporal' and type != 'spatial-temporal':
            raise AssertionError('type can only be "temporal" or "spatial-temporal"')
        self.dynamic_data = dynamic_data
        self.static_data = static_data
        self.label = label
        self.input_window = input_window
        self.output_window = output_window
        self.data_mixed = data_mixed
        self.normalization = normalization
        self.stride = stride
        self.type = type
        self.mode_dict = kwargs
        self.temporal_rate = temporal_rate
        num_positions = self.dynamic_data.shape[0]
        if spatial_indexes is None:
            self.spatial_indexes = []
            for i in range(3):
                self.spatial_indexes.append(list(range(0, num_positions)))
        else:
            self.spatial_indexes = spatial_indexes
        self.train_dynamic_data, self.train_static_data, self.train_label, self.val_dynamic_data, self.val_static_data, self.val_label, self.test_dynamic_data, self.test_static_data, self.test_label = \
            split_by_spatial_temporal(dynamic_data=self.dynamic_data, static_data=self.static_data, label=self.label, temporal_rate=self.temporal_rate, spatial_indexes=self.spatial_indexes)

        if type == 'temporal':
            # print(self.mode_dict)
            self.train_process = temporal_data_preprocess(static_data=self.train_static_data, dynamic_data=self.train_dynamic_data, label=self.train_label,
                                                        input_window=self.input_window, output_window=self.output_window, stride=self.stride, **self.mode_dict)
            self.val_process = temporal_data_preprocess(static_data=self.val_static_data, dynamic_data=self.val_dynamic_data, label=self.val_label,
                                                      input_window=self.input_window, output_window=self.output_window, stride=self.stride, **self.mode_dict)
            self.test_process = temporal_data_preprocess(static_data=self.test_static_data, dynamic_data=self.test_dynamic_data, label=self.test_label,
                                                       input_window=self.input_window, output_window=self.output_window, stride=self.stride, **self.mode_dict)
        else:
            self.train_process = spatial_temporal_data_preprocess(static_data=self.train_static_data, dynamic_data=self.train_dynamic_data, label=self.train_label, input_window=self.input_window, output_window=self.output_window, stride=self.stride)
            self.val_process = spatial_temporal_data_preprocess(static_data=self.val_static_data, dynamic_data=self.val_dynamic_data, label=self.val_label, input_window=self.input_window, output_window=self.output_window, stride=self.stride)
            self.test_process = spatial_temporal_data_preprocess(static_data=self.test_static_data, dynamic_data=self.test_dynamic_data, label=self.test_label, input_window=self.input_window, output_window=self.output_window, stride=self.stride)

    def get_origin_data(self):
        """
        Retrieves the original data for training, validation, and testing sets.

        Returns:
        -------
        tuple of torch.Tensor
            Original data and labels for train, validation, and test sets.
        """
        train_data = self.train_process.get_origin_data()
        train_label = self.train_process.get_origin_label()
        val_data = self.val_process.get_origin_data()
        val_label = self.val_process.get_origin_label()
        test_data = self.test_process.get_origin_data()
        test_label = self.test_process.get_origin_label()
        return train_data, train_label, val_data, val_label, test_data, test_label
        
    def get_dl_dataset(self):
        """
        Prepares the deep learning dataset for training, validation, and testing.

        Returns:
        -------
        tuple of dataloaders
            The dataloaders for train, validation, and test sets.
        """
        train_data = self.train_process.get_data()
        train_label = self.train_process.get_label()
        val_data = self.val_process.get_data()
        val_label = self.val_process.get_label()
        test_data = self.test_process.get_data()
        test_label = self.test_process.get_label()
        trainSet = dl_dataset(train_data, train_label)
        valSet = dl_dataset(val_data, val_label)
        testSet = dl_dataset(test_data, test_label)
        return trainSet, valSet, testSet

    # Get mearching learning data of xgboost
    def get_ml_dataset(self):
        """
        Prepares the machine learning dataset for temporal models like xgboost.

        Returns:
        -------
        tuple of torch.Tensor
            The machine learning datasets for train, validation, and test sets.

        Raises:
        ------
        AssertionError
            If the type is not 'temporal'.
        """
        if self.type != 'temporal':
            raise AssertionError("Ml models only support temporal data!")
        train_data = self.train_process.get_data()
        train_data = train_data.view(train_data.shape[0], -1)
        # print(f'train_data.shape is {train_data.shape}')
        train_label = self.train_process.get_label()
        val_data = self.val_process.get_data()
        val_data = val_data.view(val_data.shape[0], -1)
        val_label = self.val_process.get_label()
        test_data = self.test_process.get_data()
        test_data = test_data.view(test_data.shape[0], -1)
        test_label = self.test_process.get_label()
        trainSet = ml_dataset(train_data, train_label)
        valSet = ml_dataset(val_data, val_label)
        testSet = ml_dataset(test_data, test_label)
        return trainSet, valSet, testSet

    # Get Mearchine learning data of SEIR、SIR、ARIMA
    def get_ml_dataset_without_divide(self):
        """
        Prepares the machine learning dataset without dividing by temporal features
        for models like SEIR, SIR, ARIMA.

        Returns:
        -------
        tuple of torch.Tensor
            The datasets for train, validation, and test sets.

        Raises:
        ------
        AssertionError
            If the type is not 'temporal'.
        """
        if self.type != 'temporal':
            raise AssertionError("Ml models only support temporal data!")
        train_data = self.train_process.get_ml_data()
        train_label = self.train_process.get_ml_label()
        val_data = self.val_process.get_ml_data()
        val_label = self.val_process.get_ml_label()
        test_data = self.test_process.get_ml_data()
        test_label = self.test_process.get_ml_label()
        trainSet = ml_dataset(train_data, train_label)
        valSet = ml_dataset(val_data, val_label)
        testSet = ml_dataset(test_data, test_label)
        return trainSet, valSet, testSet

    def get_features_num(self):
        """
        Retrieves the number of features in the datasets.

        Returns:
        -------
        int
            The number of features.

        Raises:
        ------
        AssertionError
            If the number of features is not consistent across datasets.
        """
        train_features_num = self.train_process.get_data().shape[-1]
        val_features_num = self.val_process.get_data().shape[-1]
        test_features_num = self.test_process.get_data().shape[-1]
        if train_features_num != val_features_num or val_features_num != test_features_num:
            raise AssertionError("the numbers of features are not equal in train set, validation set and test set!")
        return train_features_num

    def get_position_num(self):
        """
        Retrieves the number of positions for spatial-temporal data.

        Returns:
        -------
        int
            The number of positions.

        Raises:
        ------
        AssertionError
            If the type is not 'spatial-temporal'.
        """
        if self.type != 'spatial-temporal':
            raise AssertionError("Only spatial-temporal models need the number of positions")
        return self.train_process.get_data().shape[1]

    def reverse_test_norm(self, label, data= None):
        """
        Reverses the normalization for test data.

        Parameters:
        ----------
        label : torch.Tensor
            The label tensor to reverse normalize.
        data : torch.Tensor, optional
            The data tensor to reverse normalize, by default None.

        Returns:
        -------
        torch.Tensor or tuple of torch.Tensor
            The reverse normalized label and, if provided, data.
        """
        return self.test_process.reverse_normalization(label, data)

    def get_static_dim(self):
        """
        Retrieves the dimensionality of the static data.

        Returns:
        -------
        int
            The dimensionality of the static data.
        """
        return self.static_data.shape[-1]
    
    def get_dynamic_dim(self):
        """
        Retrieves the dimensionality of the dynamic data.

        Returns:
        -------
        int
            The dimensionality of the dynamic data.
        """
        return self.dynamic_data.shape[-1]
        