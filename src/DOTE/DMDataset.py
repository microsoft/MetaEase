import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# dataset definition
class DmDataset(Dataset):
    def __init__(self, props=None, env=None, is_test=None):
        # store the inputs and outputs
        assert props != None and env != None and is_test != None

        num_nodes = env.get_num_nodes()
        env.test(is_test)
        tms = env._simulator._cur_hist._tms
        opts = env._simulator._cur_hist._opts
        tms = [np.asarray([tms[i]]) for i in range(len(tms))]

        np_tms = np.vstack(tms)
        np_tms = np_tms.T
        np_tms_flat = np_tms.flatten('F')
        
        # print(len(tms), len(opts))
        assert (len(tms) == len(opts))
        X_ = []
        for histid in range(len(tms) - props.hist_len):
            start_idx = histid * num_nodes * (num_nodes - 1)
            end_idx = start_idx + props.hist_len * num_nodes * (num_nodes - 1)
            X_.append(np_tms_flat[start_idx:end_idx])

        self.X = np.asarray(X_)
        self.y = np.asarray([np.append(tms[i], opts[i]) for i in range(props.hist_len, len(opts))])


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]