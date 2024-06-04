import torch


def split_by_spatio_temporal(
    dynamic_data,
    static_data,
    label,
    temporal_rate,
    spatio_indexes
):
    dynamic_data = torch.Tensor(dynamic_data)
    static_data = torch.Tensor(static_data)
    label = torch.Tensor(label)
    assert isinstance(temporal_rate, list) or isinstance(temporal_rate, tuple), "temporal_rate's type must be tuple or list"
    assert isinstance(spatio_indexes, list) or isinstance(spatio_indexes, tuple), "temporal_rate's type must be tuple or list"
    assert len(temporal_rate) == 3 and len(spatio_indexes) == 3, "time_rate's length and spatio_indexes's length must be 3"
    assert temporal_rate[0] + temporal_rate[1] + temporal_rate[2] == 1, "the sum of 3 time_rates must be 1"
    assert temporal_rate[0] != 0 and temporal_rate[1] != 0 and temporal_rate[2] != 0, "all the item of temporal_rate can't be 0"
    train_spatio_indexes = list(set(spatio_indexes[0]))
    val_spatio_indexes = list(set(spatio_indexes[1]))
    test_spatio_indexes = list(set(spatio_indexes[2]))
    loc_num = dynamic_data.shape[0]
    total_temporal_num = dynamic_data.shape[1]
    for i in train_spatio_indexes:
        if not isinstance(i, int):
            raise AssertionError("the item in spatio_indexes must be int!")
        elif i < 0 or i >= loc_num:
            raise AssertionError("the item in spatio_indexes must be between 0 and loc_num - 1!")

    for i in val_spatio_indexes:
        if not isinstance(i, int):
            raise AssertionError("the item in spatio_indexes must be int!")
        elif i < 0 or i >= loc_num:
            raise AssertionError("the item in spatio_indexes must be between 0 and loc_num - 1!")
        
    for i in test_spatio_indexes:
        if not isinstance(i, int):
            raise AssertionError("the item in spatio_indexes must be int!")
        elif i < 0 or i >= loc_num:
            raise AssertionError("the item in spatio_indexes must be between 0 and loc_num - 1!")

    train_temporal_num = int(total_temporal_num * temporal_rate[0])
    val_temporal_num = int(total_temporal_num * temporal_rate[1])
    test_temporal_num = total_temporal_num - train_temporal_num - val_temporal_num

    train_dynamic_data_spatio_tmp = dynamic_data[:, 0: train_temporal_num, :]
    val_dynamic_data_spatio_tmp = dynamic_data[:, train_temporal_num: train_temporal_num + val_temporal_num, :]
    test_dynamic_data_spatio_tmp = dynamic_data[:, train_temporal_num + val_temporal_num: total_temporal_num, :]
    # print(f'train_dynamic_data_spatio_tmp.shape is {train_dynamic_data_spatio_tmp.shape}, val_dynamic_data_spatio_tmp.shape is {val_dynamic_data_spatio_tmp.shape}, test_dynamic_data_spatio_tmp.shape is {test_dynamic_data_spatio_tmp.shape}')

    train_label_spatio_tmp = label[:, 0: train_temporal_num, :]
    val_label_spatio_tmp = label[:, train_temporal_num: train_temporal_num + val_temporal_num, :]
    test_label_spatio_tmp = label[:, train_temporal_num + val_temporal_num: total_temporal_num, :]

    train_dynamic_data = torch.Tensor([])
    val_dynamic_data = torch.Tensor([])
    test_dynamic_data = torch.Tensor([])

    train_static_data = torch.Tensor([])
    val_static_data = torch.Tensor([])
    test_static_data = torch.Tensor([])

    train_label = torch.Tensor([])
    val_label = torch.Tensor([])
    test_label = torch.Tensor([])

    for i in spatio_indexes[0]:
        # print(f'train_index is {i}')
        train_dynamic_data = torch.cat((train_dynamic_data, torch.unsqueeze(train_dynamic_data_spatio_tmp[i], 0)),0)
        train_label = torch.cat((train_label, torch.unsqueeze(train_label_spatio_tmp[i], 0)), 0)
        train_static_data = torch.cat((train_static_data, torch.unsqueeze(static_data[i], 0)), 0)
        # print(f'train_dynamic_data.shape is {torch.Tensor(train_dynamic_data).shape}')
        # print(f'train_static_data.shape is {torch.Tensor(train_static_data).shape}')
    for i in spatio_indexes[1]:
        # print(f'val_index is {i}')
        val_dynamic_data = torch.cat((val_dynamic_data, torch.unsqueeze(val_dynamic_data_spatio_tmp[i],0)),0)
        val_label = torch.cat((val_label, torch.unsqueeze(val_label_spatio_tmp[i],0)),0)
        val_static_data = torch.cat((val_static_data, torch.unsqueeze(static_data[i],0)), 0)
        # print(f'val_dynamic_data.shape is {torch.Tensor(val_dynamic_data).shape}')
        # print(f'val_static_data.shape is {torch.Tensor(val_static_data).shape}')
    for i in spatio_indexes[2]:
        # print(f'test_index is {i}')
        test_dynamic_data = torch.cat((test_dynamic_data, torch.unsqueeze(test_dynamic_data_spatio_tmp[i],0)),0)
        test_label = torch.cat((test_label, torch.unsqueeze(test_label_spatio_tmp[i],0)),0)
        test_static_data = torch.cat((test_static_data, torch.unsqueeze(static_data[i],0)), 0)
        # print(f'test_dynamic_data.shape is {torch.Tensor(test_dynamic_data).shape}')
        # print(f'test_static_data.shape is {torch.Tensor(test_static_data).shape}')

    return train_dynamic_data, train_static_data, train_label,val_dynamic_data, val_static_data, val_label, test_dynamic_data, test_static_data, test_label


if __name__ == "__main__":
    static_data = torch.ones((20,10)) #
    dynamic_data = torch.ones((20,60,15))
    label = torch.ones((20,60,1))
    spatio_index1 = list(range(0,12))
    spatio_index2 = list(range(12,16))
    spatio_index3 = list(range(16,20))
    split_by_spatio_temporal(dynamic_data=dynamic_data,static_data=static_data,label=label,temporal_rate=[0.6,0.2,0.2],spatio_indexes=[spatio_index1, spatio_index2, spatio_index3])