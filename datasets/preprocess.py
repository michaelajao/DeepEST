import torch
from typing import Optional
from .dataspliter import split_by_spatio_temporal
from .dataset import DlDataset, MlDataset


def merge_timestep(
    dynamic_data,
    label,
    merge_len: int = 1,
    mode: str = "sum",
) -> torch.Tensor:
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


class TemporalDataPreprocess():
    '''
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
    '''
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

    def getData(self):
        return self.data

    def getLabel(self):
        return self.label

    def getMlData(self):
        return self.origin_data

    def getMlLabel(self):
        return self.origin_label
    
    def getOriginData(self):
        return self.origin_data

    def getOriginLabel(self):
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


class SpatioTemporalDataPreprocess():
    '''
    kwargs: define the mode of how to choose geometirc information for
    temporal-only models, the format of kwargs is
    {
        mode:0,
        method:'sum' or 'mean'
    }, which means in mode 0, we will use the features of all regions. The
    specific way to use is: sum or average the features
    {
        mode:1
        place:0 -> N
    }, which means in mode 1, we only use the features of a certain place,
    which is specified by the place parameter,
    which ranges from 0 to N-1. Where N is the total number of places
    '''
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

    def getData(self):
        return self.data

    def getLabel(self):
        return self.label

    def getOriginData(self):
        return self.origin_data

    def getOriginLabel(self):
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

class PreprocessData():
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
        type: str = 'temporal' or 'spatio-temporal',
        temporal_rate: list = [0.6, 0.2, 0.2],
        spatio_indexes: list = None,
        **kwargs: Optional[dict],
    ):
        if type != 'temporal' and type != 'spatio-temporal':
            raise AssertionError('type can only be "temporal" or "spatio-temporal"')
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
        if spatio_indexes is None:
            self.spatio_indexes = []
            for i in range(3):
                self.spatio_indexes.append(list(range(0, num_positions)))
        else:
            self.spatio_indexes = spatio_indexes
        self.train_dynamic_data, self.train_static_data, self.train_label, self.val_dynamic_data, self.val_static_data, self.val_label, self.test_dynamic_data, self.test_static_data, self.test_label = \
            split_by_spatio_temporal(dynamic_data=self.dynamic_data, static_data=self.static_data, label=self.label, temporal_rate=self.temporal_rate, spatio_indexes=self.spatio_indexes)

        if type == 'temporal':
            # print(self.mode_dict)
            self.train_process = TemporalDataPreprocess(static_data=self.train_static_data, dynamic_data=self.train_dynamic_data, label=self.train_label,
                                                        input_window=self.input_window, output_window=self.output_window, stride=self.stride, **self.mode_dict)
            self.val_process = TemporalDataPreprocess(static_data=self.val_static_data, dynamic_data=self.val_dynamic_data, label=self.val_label,
                                                      input_window=self.input_window, output_window=self.output_window, stride=self.stride, **self.mode_dict)
            self.test_process = TemporalDataPreprocess(static_data=self.test_static_data, dynamic_data=self.test_dynamic_data, label=self.test_label,
                                                       input_window=self.input_window, output_window=self.output_window, stride=self.stride, **self.mode_dict)
        else:
            self.train_process = SpatioTemporalDataPreprocess(static_data=self.train_static_data, dynamic_data=self.train_dynamic_data, label=self.train_label, input_window=self.input_window, output_window=self.output_window, stride=self.stride)
            self.val_process = SpatioTemporalDataPreprocess(static_data=self.val_static_data, dynamic_data=self.val_dynamic_data, label=self.val_label, input_window=self.input_window, output_window=self.output_window, stride=self.stride)
            self.test_process = SpatioTemporalDataPreprocess(static_data=self.test_static_data, dynamic_data=self.test_dynamic_data, label=self.test_label, input_window=self.input_window, output_window=self.output_window, stride=self.stride)

    def getOriginData(self):
        train_data = self.train_process.getOriginData()
        train_label = self.train_process.getOriginLabel()
        val_data = self.val_process.getOriginData()
        val_label = self.val_process.getOriginLabel()
        test_data = self.test_process.getOriginData()
        test_label = self.test_process.getOriginLabel()
        return train_data, train_label, val_data, val_label, test_data, test_label
        
    def getDlDataSet(self):
        train_data = self.train_process.getData()
        train_label = self.train_process.getLabel()
        val_data = self.val_process.getData()
        val_label = self.val_process.getLabel()
        test_data = self.test_process.getData()
        test_label = self.test_process.getLabel()
        trainSet = DlDataset(train_data, train_label)
        valSet = DlDataset(val_data, val_label)
        testSet = DlDataset(test_data, test_label)
        return trainSet, valSet, testSet

    # Get mearching learning data of xgboost
    def getMlDataSet(self):
        if self.type != 'temporal':
            raise AssertionError("Ml models only support temporal data!")
        train_data = self.train_process.getData()
        train_data = train_data.view(train_data.shape[0], -1)
        # print(f'train_data.shape is {train_data.shape}')
        train_label = self.train_process.getLabel()
        val_data = self.val_process.getData()
        val_data = val_data.view(val_data.shape[0], -1)
        val_label = self.val_process.getLabel()
        test_data = self.test_process.getData()
        test_data = test_data.view(test_data.shape[0], -1)
        test_label = self.test_process.getLabel()
        trainSet = MlDataset(train_data, train_label)
        valSet = MlDataset(val_data, val_label)
        testSet = MlDataset(test_data, test_label)
        return trainSet, valSet, testSet

    # Get Mearchine learning data of SEIR、SIR、ARIMA
    def getMlDataSetWithoutDivide(self):
        if self.type != 'temporal':
            raise AssertionError("Ml models only support temporal data!")
        train_data = self.train_process.getMlData()
        train_label = self.train_process.getMlLabel()
        val_data = self.val_process.getMlData()
        val_label = self.val_process.getMlLabel()
        test_data = self.test_process.getMlData()
        test_label = self.test_process.getMlLabel()
        trainSet = MlDataset(train_data, train_label)
        valSet = MlDataset(val_data, val_label)
        testSet = MlDataset(test_data, test_label)
        return trainSet, valSet, testSet

    def getFeaturesNum(self):
        train_features_num = self.train_process.getData().shape[-1]
        val_features_num = self.val_process.getData().shape[-1]
        test_features_num = self.test_process.getData().shape[-1]
        if train_features_num != val_features_num or val_features_num != test_features_num:
            raise AssertionError("the numbers of features are not equal in train set, validation set and test set!")
        return train_features_num

    def getPositionNum(self):
        if self.type != 'spatio-temporal':
            raise AssertionError("Only spatio-temporal models need the number of positions")
        return self.train_process.getData().shape[1]

    def reverse_test_norm(self, label, data= None):
        return self.test_process.reverse_normalization(label, data)
