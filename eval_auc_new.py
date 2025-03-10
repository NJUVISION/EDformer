import dv_processing as dv
import random
import torch
import pandas as pd
import numpy as np
import argparse
from model import EDformer as EDformer
from metrics import EventStructuralRatio
from emlb_dataset import Dataset
import time
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc

TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3
LABEL_COLUMN = 4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Inference(object):
    def __init__(self, model, seq_len) -> None:
        self.model = model
        self.seq_len = seq_len

    def inference(self, event_array):
        num_samples = len(event_array) // self.seq_len
        
        min_t = np.min(event_array[:, TIMESTAMP_COLUMN])
        max_t = np.max(event_array[:, TIMESTAMP_COLUMN])
        
        res = dv.EventStore()

        x = event_array[:, X_COLUMN]
        y = event_array[:, Y_COLUMN]
        polarity = event_array[:, POLARITY_COLUMN]
        timestamp = self.normalize_column(
            event_array[:, TIMESTAMP_COLUMN])
        event_array = np.hstack((timestamp.reshape(-1, 1), x.reshape(-1, 1),
                                 y.reshape(-1, 1), polarity.reshape(-1, 1)))
        event_array_reshaped = event_array[:num_samples *
                                           self.seq_len, :].reshape((num_samples, self.seq_len, 4))
        
        label_pred = []
        
        for i in range(num_samples):
            events_slice = event_array_reshaped[i, :, :]
            res, label_filter_stacked =  self.process_slice(events_slice, res, min_t, max_t)
            label_pred.append(label_filter_stacked)

        label_pred_stacked = np.vstack(label_pred)
        
        return res, label_pred_stacked

    def normalize_column(self, column):
        min_val = np.min(column)
        max_val = np.max(column)
        normalized_column = (column - min_val) / (max_val - min_val)
        return normalized_column

    def process_slice(self, events_slice, res, min_t, max_t):
        num = 1
        states = None
        memories = None
        events_filter = []
        label_filter = []
        processed_events = torch.tensor(events_slice).reshape(
            (1, events_slice.shape[0], events_slice.shape[1])).to(dtype=torch.float32).cuda()
        sub_sequence_size = self.seq_len // num
        for j in range(num):
            start_idx = j * sub_sequence_size
            end_idx = (j + 1) * sub_sequence_size
            sub_sequence = processed_events[:, start_idx:end_idx, :]
            with torch.no_grad():
                start_time = time.time()
                f = self.model(sub_sequence)
                end_time = time.time()
            # print("time cost:", float(end_time - start_time) * 1000.0, "ms")
            predictions = torch.sigmoid(f)
            
            predictions_np = predictions.cpu().numpy()
            label_filter.append(predictions_np)
            
            indices = np.where(predictions.cpu() == 0)[1]
            events_filter.append(sub_sequence.squeeze(0)
                                 [indices].cpu().numpy())
        
        label_filter_stacked = np.vstack(label_filter)
            
        events_filter_stacked = np.vstack(events_filter)
        events_filter_stacked[:,TIMESTAMP_COLUMN] = events_filter_stacked[:,TIMESTAMP_COLUMN] * (max_t - min_t) + min_t
        sorted_indices = np.argsort(events_filter_stacked[:, TIMESTAMP_COLUMN])
        events_filter_stacked = events_filter_stacked[sorted_indices]

        for j in range(events_filter_stacked.shape[0]):
            timestamp, x, y, polarity = events_filter_stacked[j, 0], events_filter_stacked[j,
                                                                                           1], events_filter_stacked[j, 2], events_filter_stacked[j, 3]
            timestamp = int(timestamp)
            x = int(x)
            y = int(y)
            polarity = bool(polarity)
            res.push_back(timestamp, x, y, polarity)

        return res, label_filter_stacked

def normalize_column(column):
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

def process_dataset(model, dataset_path, dataset_name, hz, writer):
    print(f"处理数据集: {dataset_name}, 频率: {hz}Hz")
    
    event_file = dataset_path
    
    try:
        events = pd.read_csv(event_file, skiprows=1, delimiter=' ', dtype={
            'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8})
        events = events.values
        
        print(f"数据形状: {events.shape}")
        
        seq_len = 4096
        inference = Inference(model, seq_len)
        res, label_pred_stacked = inference.inference(events)
        
        num_samples = len(events) // seq_len
        event_tmp = events[:num_samples*seq_len]
        event_label = event_tmp[:,LABEL_COLUMN]
        label_pred_stacked = label_pred_stacked.reshape(-1,1)
        
        fpr, tpr, thresholds = roc_curve(event_label, label_pred_stacked)
        roc_auc = auc(fpr, tpr)
        
        print(f"{dataset_name} {hz}Hz ROC AUC = {roc_auc:.4f}")
        
        writer.add_scalar(f"{dataset_name}/AUC", roc_auc, hz)
        
        writer.add_figure(
            f"{dataset_name}/ROC_Curve_{hz}Hz",
            plot_roc_curve(fpr, tpr, roc_auc, f"{dataset_name} {hz}Hz"),
            hz
        )
        
        return roc_auc
    except Exception as e:
        print(f"处理 {dataset_name} {hz}Hz 时出错: {e}")
        return None

def plot_roc_curve(fpr, tpr, roc_auc, title):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DND21 datasets.')
    parser.add_argument('-b', '--base_path',  type=str,
                        default='/workspace/shared/event_dataset/ECCV2024_datasets/AUC_test', 
                        help='基础数据路径')
    parser.add_argument('-o', '--output_path', type=str,
                        default='./results', help='输出结果路径')
    parser.add_argument('-m', '--model_path', type=str,
                        default='./pretrained_model.pth', help='模型路径')
    parser.add_argument('-l', '--log_dir', type=str,
                        default='./runs/event_denoising', help='TensorBoard日志目录')
    args = parser.parse_args()

    setup_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    writer = SummaryWriter(args.log_dir)

    print(f"加载模型: {args.model_path}")
    mod = EDformer().to(device)
    mod.load_state_dict(torch.load(args.model_path, map_location=device))
    mod.eval()

    hz_list = [1, 3, 5, 7, 10]
    datasets = [
        {"name": "driving_mix", "filename": "driving_mix_result.txt"},
        {"name": "mix", "filename": "mix_result.txt"}
    ]
    
    all_results = {}
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        filename = dataset["filename"]
        all_results[dataset_name] = {}
        
        for hz in hz_list:
            dataset_path = os.path.join(args.base_path, f"{hz}hz", filename)
            
            if os.path.exists(dataset_path):
                auc_value = process_dataset(mod, dataset_path, dataset_name, hz, writer)
                all_results[dataset_name][hz] = auc_value
            else:
                print(f"警告: 文件不存在 - {dataset_path}")
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    for dataset_name, results in all_results.items():
        hz_values = []
        auc_values = []
        
        for hz, auc_value in sorted(results.items()):
            if auc_value is not None:
                hz_values.append(hz)
                auc_values.append(auc_value)
        
        plt.plot(hz_values, auc_values, marker='o', label=dataset_name)
    
    plt.xlabel('频率 (Hz)')
    plt.ylabel('AUC')
    plt.title('不同频率下各数据集的AUC值')
    plt.grid(True)
    plt.legend()
    
    summary_fig_path = os.path.join(args.output_path, 'auc_summary.png')
    plt.savefig(summary_fig_path)
    
    writer.add_figure('Summary/AUC_vs_Hz', plt.gcf())
    
    writer.close()
    
    print(f"所有结果已保存到 {args.output_path}")
    print(f"TensorBoard日志已保存到 {args.log_dir}")
    print(f"可以使用以下命令查看TensorBoard: tensorboard --logdir={args.log_dir}")