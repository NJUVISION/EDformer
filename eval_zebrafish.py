import os
import argparse
import numpy as np
import torch
import random
import dv_processing as dv
import cv2
from datetime import timedelta
from emlb_dataset import Dataset
from metrics import EventStructuralRatio
from model import EDformer as EDformer

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

    def inference(self, events):
        res = dv.EventStore()
        event_array = np.empty((len(events), 4))

        for i, e in enumerate(events):
            event_array[i, 0] = e.timestamp()
            event_array[i, 1] = e.x()
            event_array[i, 2] = e.y()
            event_array[i, 3] = e.polarity()
            
        min_t = np.min(event_array[:, TIMESTAMP_COLUMN])
        max_t = np.max(event_array[:, TIMESTAMP_COLUMN])

        print(event_array.shape)
        
        x = event_array[:, X_COLUMN]
        y = event_array[:, Y_COLUMN]
        polarity = event_array[:, POLARITY_COLUMN]
        timestamp = self.normalize_column(
            event_array[:, TIMESTAMP_COLUMN])
        event_array = np.hstack((timestamp.reshape(-1, 1), x.reshape(-1, 1),
                            y.reshape(-1, 1), polarity.reshape(-1, 1)))
        
        num_samples = len(events) // self.seq_len
        event_array_reshaped = event_array[:num_samples *
                                           self.seq_len, :].reshape((num_samples, self.seq_len, 4))

        for i in range(num_samples):
            events_slice = event_array_reshaped[i, :, :]
            self.process_slice(events_slice, res, min_t, max_t)
            
        return res

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
            predictions[predictions >= 0.85] = 1
            predictions[predictions < 0.85] = 0
            indices = np.where(predictions.cpu() == 0)[1]
            events_filter.append(sub_sequence.squeeze(0)
                                 [indices].cpu().numpy()) 
        events_filter_stacked = np.vstack(events_filter)

        for j in range(events_filter_stacked.shape[0]):
            timestamp, x, y, polarity = events_filter_stacked[j, 0], events_filter_stacked[j,
                                                                                           1], events_filter_stacked[j, 2], events_filter_stacked[j, 3]
            timestamp = int(timestamp * (max_t - min_t) + min_t)
            x = int(x)
            y = int(y)
            polarity = bool(polarity)
            res.push_back(timestamp, x, y, polarity)

        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run zebrafish datasets.')
    parser.add_argument('-i', '--input_path',  type=str,
                        default='/workspace/shared/event_dataset/ED24/zebra_fish/blood_vessels_1.aedat4', help='path to load dataset')
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

    reader = dv.io.MonoCameraRecording(event_file)

    events = dv.EventStore()
    
    i = 0

    while reader.isRunning():
        tmp = reader.getNextEventBatch()
        if tmp is not None:
            events.add(tmp)
            
    model, seq_len = mod, 4096
    inference = Inference(model, seq_len)
    res = inference.inference(events)
    
    res_array = np.empty((len(res), 4))

    for i, e in enumerate(res):
        res_array[i, 0] = e.timestamp()
        res_array[i, 1] = e.x()
        res_array[i, 2] = e.y()
        res_array[i, 3] = e.polarity()
    
    resolution = reader.getEventResolution()
    metric = EventStructuralRatio(resolution)
    
    score_before = metric.evalEventStorePerNumber(events)

    score = metric.evalEventStorePerNumber(res)
    
    mean_score = np.mean(score[~np.isnan(score)])
    
    print(event_file, ' before: ', score_before.mean(), ' after: ', mean_score)
    
    visualizer = dv.visualization.EventVisualizer(resolution)
    visualizer.setBackgroundColor(dv.visualization.colors.white())
    visualizer.setPositiveColor(dv.visualization.colors.red())
    visualizer.setNegativeColor(dv.visualization.colors.blue())
    
    frame_counter = 0
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    def slicing_callback(events: dv.EventStore):
        global frame_counter
        frame = visualizer.generateImage(events)
        filename = os.path.join(args.output_path, f'frame_{frame_counter}.jpg')
        cv2.imwrite(filename, frame)
        frame_counter += 1
    
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(timedelta(milliseconds=100), slicing_callback)
    slicer.accept(res)
    
    
    
