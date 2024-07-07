import dv_processing as dv
import random
import torch
import pandas as pd
import numpy as np
import argparse
from model import EDformer as EDformer
from metrics import EventStructuralRatio
from emlb_dataset import Dataset

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
                f = mod(sub_sequence)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DND21 datasets.')
    parser.add_argument('-i', '--input_path',  type=str,
                        default='/workspace/shared/event_dataset/ECCV2024_datasets/AUC_test/5hz/driving_mix_result.txt', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str,
                        default='./results', help='path to output dataset')
    parser.add_argument('-m', '--model_path', type=str,
                        default='./pretrained_model.pth', help='path to model')
    args = parser.parse_args()

    setup_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0") 

    mod = EDformer().cuda()
    mod.load_state_dict(torch.load(args.model_path, map_location=device))

    mod.eval()

    class_scores = {}

    event_file = args.input_path

    events = pd.read_csv(event_file, skiprows=1, delimiter=' ', dtype={
        'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8})
    events = events.values
    
    print(events)
    print(events.shape)
    
    metric = EventStructuralRatio((346, 260))

    model, seq_len = mod, 4096
    inference = Inference(model, seq_len)
    res, label_pred_stacked = inference.inference(events)
    
    num_samples = len(events) // 4096
    event_tmp = events[:num_samples*4096]
    event_label = event_tmp[:,LABEL_COLUMN]
    label_pred_stacked = label_pred_stacked.reshape(-1,1)
    
    print(label_pred_stacked.shape, event_label.shape)
    
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(event_label, label_pred_stacked)
    
    roc_auc = auc(fpr, tpr)
    
    print(f'ROC = {roc_auc}')