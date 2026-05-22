
"""
    1. [推荐] 联合训练 (使用包含多个 CSV 文件的文件夹):
    python train_actuator_network.py --mode train --data my_robot/logs_folder --output my_robot/actuator_net.pt

    2. 单文件训练:
    python train_actuator_network.py --mode train --data my_robot/motor_log.csv --output my_robot/actuator_net.pt
    
    3. 评估与画图查看 (Play 模式，不训练，仅加载现有模型对比真实曲线):
    python train_actuator_network.py --mode play --data my_robot/logs_folder --output my_robot/actuator_net.pt
"""

import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pandas as pd
import glob

BASE_PATH = os.path.join(os.path.dirname(__file__))

class Config:
    def __init__(self):
        self.lr = 8e-4  # 8e-4
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.epochs = 200
        self.batch_size = 128
        self.device = "cuda:0"
        self.in_dim = 6
        self.units = 32 # 32
        self.layers = 2  # 2
        self.out_dim = 1
        self.act = "softsign"
        self.dt = 0.02
        
        # 指定要训练的关节
        self.train_indices = [0, 1, 2, 3, 4, 5] 

        self.pos_scale = 1.0   
        self.vel_scale = 1.0    
        self.torque_scale = 1.0

class ActuatorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["joint_states"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

class Act(nn.Module):
    def __init__(self, act, slope=0.05):
        super(Act, self).__init__()
        self.act = act
        self.slope = slope
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, input):
        if self.act == "relu":
            return F.relu(input)
        elif self.act == "leaky_relu":
            return F.leaky_relu(input)
        elif self.act == "sp":
            return F.softplus(input, beta=1.0)
        elif self.act == "leaky_sp":
            return F.softplus(input, beta=1.0) - self.slope * F.relu(-input)
        elif self.act == "elu":
            return F.elu(input, alpha=1.0)
        elif self.act == "leaky_elu":
            return F.elu(input, alpha=1.0) - self.slope * F.relu(-input)
        elif self.act == "ssp":
            return F.softplus(input, beta=1.0) - self.shift
        elif self.act == "leaky_ssp":
            return (
                F.softplus(input, beta=1.0) - self.slope * F.relu(-input) - self.shift
            )
        elif self.act == "tanh":
            return torch.tanh(input)
        elif self.act == "leaky_tanh":
            return torch.tanh(input) + self.slope * input
        elif self.act == "swish":
            return torch.sigmoid(input) * input
        elif self.act == "softsign":
            return F.softsign(input)
        else:
            raise RuntimeError(f"Undefined activation called {self.act}")

def build_mlp(config):
    mods = [nn.Linear(config.in_dim, config.units), Act(config.act)]
    for i in range(config.layers - 1):
        mods +=[nn.Linear(config.units, config.units), Act(config.act)]
    mods +=[nn.Linear(config.units, config.out_dim)]
    return nn.Sequential(*mods)

def load_data(data_path):
    data = pd.read_csv(data_path)
    if len(data) < 1:
        return None, 0

    num_motors = sum(1 for col in data.columns if col.startswith("tau_est_"))
    columns =["tau_est_", "tau_cal_", "joint_pos_", "joint_pos_target_", "joint_vel_"]

    data_dict = {col:[] for col in columns}
    for col in columns:
        for i in range(num_motors):
            data_dict[col].append(data[f"{col}{i}"].values)

    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key]).T

    return data_dict, num_motors

def process_data(data_dict, num_motors, step, config):

    joint_position_errors = data_dict["joint_pos_target_"] - data_dict["joint_pos_"]
    joint_velocities = data_dict["joint_vel_"]
    tau_ests = data_dict["tau_est_"]

    # 缩放
    joint_position_errors = torch.tensor(joint_position_errors, dtype=torch.float) * config.pos_scale
    joint_velocities = torch.tensor(joint_velocities, dtype=torch.float) * config.vel_scale
    tau_ests = torch.tensor(tau_ests, dtype=torch.float) / config.torque_scale

    xs, ys = [], []
    valid_indices = []
    
    for i in config.train_indices:
        if i >= num_motors:
            print(f"Warning: Index {i} out of range (max {num_motors-1})")
            continue
            
        valid_indices.append(i)

        xs_joint = [
            joint_position_errors[step:    , i:i+1],
            joint_position_errors[step-1:-1, i:i+1],
            joint_position_errors[step-2:-2, i:i+1],
            joint_velocities[step:    , i:i+1],
            joint_velocities[step-1:-1, i:i+1],
            joint_velocities[step-2:-2, i:i+1],
        ]
        tau_ests_joint = tau_ests[step:    , i:i+1]

        xs_joint = torch.cat(xs_joint, dim=1)
        xs.append(xs_joint)
        ys.append(tau_ests_joint)

    if not xs:
        return None, None, []

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    
    return xs, ys, valid_indices

def load_and_process_folder(folder_path, step, config):
    all_xs = []
    all_ys =[]
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if len(csv_files) == 0:
        print(f"Error: No CSV files found in directory {folder_path}!")
        return None, None
        
    print(f"Found {len(csv_files)} CSV files. Start processing...")
    global_num_motors = None
    
    for file_path in csv_files:
        data_dict, num_motors = load_data(file_path)
        if data_dict is None:
            continue

        if len(data_dict["tau_est_"]) <= step:
            print(f"Warning: File {os.path.basename(file_path)} is too short. Skipping...")
            continue
            
        if global_num_motors is None:
            global_num_motors = num_motors
        elif global_num_motors != num_motors:
            print(f"Warning: Motor count mismatch in {file_path}. Skipping...")
            continue
            
        xs, ys, _ = process_data(data_dict, num_motors, step, config)
        
        all_xs.append(xs)
        all_ys.append(ys)
        print(f" - Loaded {os.path.basename(file_path)}: {xs.shape[0]} valid samples.")

    if not all_xs:
        return None, None

    final_xs = torch.cat(all_xs, dim=0)
    final_ys = torch.cat(all_ys, dim=0)
    
    print(f"Total merged data samples (Legs only): {final_xs.shape[0]}")
    return final_xs, final_ys

def train_actuator_network(xs, ys, actuator_network_path, config):
    num_data = xs.shape[0]
    num_train = num_data // 5 * 4
    num_test = num_data - num_train

    dataset = ActuatorDataset({"joint_states": xs, "tau_ests": ys})
    train_set, val_set = torch.utils.data.random_split(dataset,[num_train, num_test])
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    model = build_mlp(config)
    opt = Adam(model.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    model = model.to(config.device)
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        ct = 0
        for batch in train_loader:
            data = batch["joint_states"].to(config.device)
            y_pred = model(data)
            opt.zero_grad()
            y_label = batch["tau_ests"].to(config.device)
            loss = ((y_pred - y_label) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().cpu().numpy()
            ct += 1
        epoch_loss /= ct

        test_loss = 0
        mae = 0
        ct = 0
        if epoch % 1 == 0:
            with torch.no_grad():
                for batch in test_loader:
                    data = batch["joint_states"].to(config.device)
                    y_pred = model(data)
                    y_label = batch["tau_ests"].to(config.device)
                    loss = ((y_pred - y_label) ** 2).mean()
                    test_mae = (y_pred - y_label).abs().mean()
                    test_loss += loss
                    mae += test_mae
                    ct += 1
                test_loss /= ct
                mae /= ct
            print(f"epoch: {epoch} | loss: {epoch_loss:.4f} | test loss: {test_loss:.4f} | mae: {mae:.4f}")

    model_scripted = torch.jit.script(model.to("cpu"))
    model_scripted.save(actuator_network_path)
    dummy_input = torch.randn(1, 6).to("cpu")
    torch.onnx.export(model.to("cpu"), dummy_input, actuator_network_path.replace(".pt", ".onnx"))

    return model

def train_actuator_network_and_plot_predictions(data_path, actuator_network_path, load_pretrained_model=False, config=None):
    step = 2
    if load_pretrained_model:
        model = torch.jit.load(actuator_network_path).eval()
        sample_file = glob.glob(os.path.join(data_path, "*.csv"))[0] if os.path.isdir(data_path) else data_path
        data_dict_for_plot, num_motors = load_data(sample_file)
    else:
        if os.path.isdir(data_path):
            xs, ys = load_and_process_folder(data_path, step, config)
            sample_file = glob.glob(os.path.join(data_path, "*.csv"))[0]
            data_dict_for_plot, num_motors = load_data(sample_file)
        else:
            data_dict_for_plot, num_motors = load_data(data_path)
            xs, ys, _ = process_data(data_dict_for_plot, num_motors, step, config)
        
        if xs is None: return
        model = train_actuator_network(xs, ys, actuator_network_path, config)

    # 画图验证
    xs_plot, _, trained_indices = process_data(data_dict_for_plot, num_motors, step, config)
    model = model.to("cpu")
    tau_preds = model(xs_plot).detach().reshape(len(trained_indices), -1).T * config.torque_scale
    
    plot_len = min(1000, len(tau_preds))
    timesteps = np.arange(len(tau_preds)) * config.dt
    
    cols = 2
    rows = (len(trained_indices) + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axs = axs.flatten()

    for idx, motor_idx in enumerate(trained_indices):
        axs[idx].plot(timesteps[:plot_len], data_dict_for_plot["tau_cal_"][step:step+plot_len, motor_idx], label="Calc")
        axs[idx].plot(timesteps[:plot_len], data_dict_for_plot["tau_est_"][step:step+plot_len, motor_idx], label="Real")
        axs[idx].plot(timesteps[:plot_len], tau_preds[:plot_len, idx], "--", label="Pred")
        axs[idx].set_title(f"Motor {motor_idx}")
        axs[idx].legend()

    for idx in range(len(trained_indices), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "play"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    output_path = os.path.join(BASE_PATH, args.output)

    config = Config()

    train_actuator_network_and_plot_predictions(
        data_path=data_path,
        actuator_network_path=output_path,
        load_pretrained_model=(args.mode == "play"),
        config=config,
    )

if __name__ == "__main__":
    main()
