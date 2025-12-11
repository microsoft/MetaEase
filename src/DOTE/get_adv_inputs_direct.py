import sys
import os
from collections import *
from DMDataset import *
import utils
import read_from_metaopt
# TODO: change to the abosolute gap
cwd = os.getcwd()
print(cwd)
assert "networking_envs" in cwd
sys.path.append(cwd[:cwd.find("networking_envs")] + "networking_envs")
sys.path.append(cwd[:cwd.find("networking_envs")] + "openai_baselines")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from networking_envs.networking_env.environments.ecmp.env_args_parse import parse_args
from networking_envs.networking_env.environments.ecmp import history_env
from networking_envs.networking_env.environments.consts import SOMode
from networking_envs.networking_env.utils.shared_consts import SizeConsts
from tqdm import tqdm

from datetime import datetime

# model definition
class NeuralNetworkMaxUtil(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxUtil, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits
    

class InputGeneratorSingle(nn.Module):
    def __init__(self):
        super(InputGeneratorSingle, self).__init__()
        self.input_n = nn.Parameter(torch.zeros(1, num_pairs), requires_grad=True)
        nn.init.uniform_(self.input_n, 0, 1e-7)
        self.active_n = nn.Sigmoid()
    def forward(self, x):
        x = self.input_n
        x = self.active_n(x) * 1e-5
        return x

class DOTESynthetis(nn.Module):
    def __init__(self, generator, solver_model, commodities_to_paths):
        super(DOTESynthetis, self).__init__()
        self.generator = generator
        self.generator.double()
        self.solver_model = solver_model
        self.commodities_to_paths = commodities_to_paths
        self.flatten = nn.Flatten()
        self.active_n = nn.ReLU()
    
    def normalize_sp_ratios(self, y):
        y = y + 1e-16
        paths_weight = torch.transpose(y, 0, 1)
        commodity_total_weight = self.commodities_to_paths.matmul(paths_weight)
        commodity_total_weight = 1.0 / (commodity_total_weight)
        paths_over_total = self.commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)
        paths_split = paths_weight.mul(paths_over_total)
        return torch.transpose(paths_split, 0, 1)
    
    def forward(self, x):
        inner_feature = self.flatten(self.generator(x))
        x = self.solver_model(inner_feature)
        x = self.normalize_sp_ratios(x)
        return x, inner_feature

ce_loss = torch.nn.CrossEntropyLoss()

def loss_fn_adv_maxutil(y_pred_batch, y_inner_feature, env,
                        _lagrange_multiplier_1,
                        optimal_split_ratios):
    losses = []
    loss_vals = []
    losses_3 = []
    losses_4 = []
    diff_lag_1 = []
    diff_lag_2 = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        paths_split = y_pred_batch[[i]]
        cong = compute_cong(y_inner_feature, env, paths_split)
        max_cong = torch.max(cong)
        norm_opt_split = normalize_sp_ratios(optimal_split_ratios)
        cong_opt = compute_cong(y_inner_feature, env, norm_opt_split)
        max_cong_opt = torch.max(cong_opt)

        min_split_ratio = torch.min(norm_opt_split)
        abs_opt_cong = max_cong_opt - 0.5
        norm = 1 if max_cong.item() == 0 else max_cong.item()
        loss = (-1 * max_cong + _lagrange_multiplier_1 * abs_opt_cong) / norm
        
        loss_val = max_cong / max_cong_opt.item()
        losses.append(loss)
        loss_vals.append(loss_val)
        losses_3.append(max_cong_opt)
        losses_4.append(max_cong)
        diff_lag_1.append(abs_opt_cong)
        diff_lag_2.append(-1 * min_split_ratio)
    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    ret_3 = sum(losses_3) / len(losses_3)
    ret_4 = sum(losses_4) / len(losses_4)
    
    return ret, ret_val, ret_3, ret_4,

def compute_cong(y_inner_feature, env, paths_split):
    tmp_demand_on_paths = commodities_to_paths.transpose(0,1).matmul(y_inner_feature.transpose(0,1))
    demand_on_paths = tmp_demand_on_paths.mul(paths_split.transpose(0, 1))
    flow_on_edges = paths_to_edges.transpose(0,1).matmul(demand_on_paths)
    congestion = flow_on_edges.divide(torch.tensor(np.array([env._capacities])).transpose(0,1))
    return congestion

def normalize_sp_ratios(y_pred):
    y_pred = torch.abs(y_pred)
    y_pred = y_pred + 1e-16 #eps
    paths_weight = torch.transpose(y_pred, 0, 1)
    commodity_total_weight = commodities_to_paths.matmul(paths_weight)
    commodity_total_weight = 1.0 / (commodity_total_weight)
    paths_over_total = commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)
    paths_split = paths_weight.mul(paths_over_total)
    return torch.transpose(paths_split, 0, 1)


props = parse_args(sys.argv[1:])
env = history_env.ECMPHistoryEnv(props)

ctp_coo = env._optimizer._commodities_to_paths.tocoo()
commodities_to_paths = torch.sparse_coo_tensor(np.vstack((ctp_coo.row, ctp_coo.col)), torch.DoubleTensor(ctp_coo.data), torch.Size(ctp_coo.shape))
pte_coo = env._optimizer._paths_to_edges.tocoo()
paths_to_edges = torch.sparse_coo_tensor(np.vstack((pte_coo.row, pte_coo.col)), torch.DoubleTensor(pte_coo.data), torch.Size(pte_coo.shape))

batch_size = props.so_batch_size
n_epochs = props.so_epochs
concurrent_flow_cdf = None
if props.opt_function == "MAXUTIL":
    NeuralNetwork = NeuralNetworkMaxUtil
else:
    print("Unsupported optimization function. Supported functions: MAXUTIL, MAXFLOW, MAXCOLC")
    assert False

# topo properties
num_nodes = env.get_num_nodes()
num_pairs =  num_nodes * (num_nodes - 1)
hist_len = props.hist_len

print(f"num pair = {num_pairs}, hist length = {hist_len}")


torch.manual_seed(0)
dote_model = torch.load('model_dote.pkl')

for param in dote_model.parameters():
    param.requires_grad = False

model = DOTESynthetis(InputGeneratorSingle(), dote_model, commodities_to_paths)

torch.manual_seed(1)

list_p = [p for p in model.parameters() if p.requires_grad == True]

adv_optimizer = torch.optim.Adam(list_p, 
                             lr=0.01,
                            #  weight_decay=5e-4,
                            #  betas=(0.5, 0.99),
                             )

adv_optimizer.zero_grad()

step_size = 5000
gamma = 0.5
sch_adv = torch.optim.lr_scheduler.MultiStepLR(adv_optimizer, milestones=[step_size * i for i in range(1, 1000)], gamma=gamma)

n_epochs = 500
# step_size = 10

def max_grad(grad):
    return -grad

_lagrange_multiplier_1 = torch.rand(1, requires_grad=True, dtype=torch.double)
_lagrange_multiplier_1.register_hook(max_grad) # because we maximize wrt lambda
lagrange_optimizer = torch.optim.Adam([_lagrange_multiplier_1],
                                      lr=0.01)

sch_lagrang = torch.optim.lr_scheduler.MultiStepLR(lagrange_optimizer, milestones=[step_size * i for i in range(1, 1000)], gamma=gamma)


optimal_split_ratios = torch.rand(1, env._optimizer._num_paths, requires_grad=True, dtype=torch.double)
optimal_split_ratios.double()
optimal_optimizer = torch.optim.Adam([optimal_split_ratios],
                                     lr=0.01)


sch_optimal = torch.optim.lr_scheduler.MultiStepLR(optimal_optimizer, milestones=[step_size * i for i in range(1, 1000)], gamma=gamma)

n_samples = 1
batch_size = 1
train_dataset = [(torch.randn(32, dtype=torch.double), []) for _ in range(n_samples)]
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

st_time = datetime.now()
step_size = 0.01
for i in range(100):
    for epoch in range(n_epochs):
        for (inputs, _) in train_dl:
            adv_optimizer.zero_grad()
            optimal_optimizer.zero_grad()
            lagrange_optimizer.zero_grad()
            yhat, yfeature = model(inputs)
            loss, loss_1, max_cong_opt, max_cong_dote = loss_fn_adv_maxutil(yhat, yfeature, env,
                                                                            _lagrange_multiplier_1,
                                                                            optimal_split_ratios)
            loss.backward()
            adv_optimizer.step()
            optimal_optimizer.step()
            lagrange_optimizer.step()

        if epoch % 100 == 0:
            print(f"round: {i}, epoch: {epoch} loss: {loss.item():.4f} loss_1 {loss_1.item():.4f} "
                    f"max cong opt {max_cong_opt} max cong dote {max_cong_dote} "
                    f"lr: {adv_optimizer.param_groups[-1]['lr']}")
            print(f"lagrange 1 {_lagrange_multiplier_1}")
            print(f"time: {datetime.now() - st_time}")

print(f"time: {datetime.now() - st_time}")
print("============ after training")
model.eval()

counter = 0
for (inputs, _) in train_dl:
    counter += 1
    print(f"======= sample {counter} ===========")
    yhat = model(inputs)[0]
    final_input = model.generator(inputs)
    print("input: ", final_input)
    print("input max: ", torch.max(final_input))
    print("input min: ", torch.min(final_input))
    
    if counter == 1:
        first_input = final_input.detach().numpy()
    else:
        print("differecne from first demand")
        curr_input = final_input.detach().numpy()
        diff = curr_input - first_input
        print(diff)
        print("sum diff:", np.sum(np.abs(diff)))
