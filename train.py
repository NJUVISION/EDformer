import numpy as np
import torch
import os
import random
import argparse
from model import EDformer as EDformer
from dataset import ED24
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(230086)

    parser = argparse.ArgumentParser(description='EDformer')
    parser.add_argument('root', type=str, help='dataset root path')
    parser.add_argument('N', type=int, help='number of events')
    args = parser.parse_args()

    tensorboard_log_dir = 'logs'
    models_dir = 'models'
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    writer = SummaryWriter(tensorboard_log_dir)

    device = torch.device("cuda:0")

    print('******************data loading*********************')
    dataset = ED24(args.root, args.N)
    batch_size = 96
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    model = EDformer().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    pre_val_loss = np.Inf
    pre_acc = -np.Inf
    train_accuracy = 0
    epoch = 0

    print('******************Training*********************')

    while epoch < 60:
        total_train_loss = 0
        correct_predictions = 0
        total_samples = 0

        model.train()

        for events, labels in tqdm(train_loader):
            events = events.to(device)
            labels = labels.to(device)
            f = model(events)

            optimizer.zero_grad()
            loss = criterion(f, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            predictions = torch.sigmoid(f)
            predictions[predictions >= 0.01] = 1
            predictions[predictions < 0.01] = 0
            correct_predictions += torch.sum(predictions == labels).item()
            total_samples += labels.numel()

        train_accuracy = correct_predictions / total_samples

        writer.add_scalar('Train Loss', total_train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_accuracy, epoch)

        print(f'Epoch {epoch+1}, Train Loss: {total_train_loss:.4f}, Train_Accuracy: {train_accuracy:.4f}')

        torch.save(model.state_dict(), f'models/model_{epoch}.pth')

        epoch += 1
