import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


MODEL_PARAMETERS = {
    "C1orf132": (273, 512, 6),
    "FHL2": (531, 512, 6),
    "ELOVL2_6": (531, 512, 6),
    "CCDC102B": (25, 256, 6),
    "ELOVL2_6_C1orf132": (531 + 273, 1024, 6),
    "ELOVL2_6_C1orf132_FHL2": (531 + 273 + 531, 1024, 6),
    "ELOVL2_6_C1orf132_CCDC102B": (531 + 273 + 25, 1024, 6),
    "ELOVL2_6_C1orf132_FHL2_CCDC102B": (531 + 273 + 531 + 25, 1024, 6)
}


# md = pd.read_csv("/cs/cbio/daniel/dl_methyl_to_age/data/meta_data.csv")
class CustomDataset(Dataset):
    def __init__(self, hist):
        self.all_data = hist
        self.all_data.dropna(inplace=True)
        self.tags = ['sample'] * 128
        # self.all_data = self.all_data[self.all_data.columns.drop(list(self.all_data.filter(regex='Unnamed*')))]
        # self.all_data = self.all_data[self.all_data.columns.drop(list(self.all_data.filter(regex='total_reads_origin*')))]
        # self.all_data.drop(columns=['sample_name', 'age', 'set', 'total_reads_origin'], inplace=True)
        # self.all_data.drop(columns=['sample_name'], inplace=True)
        # self.all_data = self.all_data[[c for c in self.all_data.columns if "site_" not in c and "C_count" not in c]]
        self.data = torch.tensor(self.all_data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return self.data[idx], self.tags[idx]


class Predictor(torch.nn.Module):
    def __init__(self, input_size, layer_size, layers_num=3, dropout=0.2):
        super().__init__()
        self.hidden = torch.nn.ModuleList()
        self.fc_in = torch.nn.Linear(input_size, layer_size)
        for i in range(layers_num):
            self.hidden.append(torch.nn.Linear(layer_size, layer_size))
        self.fc_out = torch.nn.Linear(layer_size, 1)
        self.drop = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.drop(self.fc_in(x)))
        for layer in self.hidden:
            x = F.relu(self.drop(layer(x)))
        x = self.fc_out(x)
        return x


def predict(marker, hist):
    device = torch.device('cpu')
    input_size, layer_size, num_layers = MODEL_PARAMETERS[marker]
    predictor = Predictor(input_size, layer_size, num_layers)
    predictor.load_state_dict(torch.load("/cs/cbio/daniel/magenet_service/predictor/models/predictor_" + marker + '_' + str(input_size)))
    predictor.to(device)
    test_data = CustomDataset(hist)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=4)
    with torch.no_grad():
        predictor.eval()
        batch_test, tags = next(iter(test_dataloader))
        out_test = predictor(batch_test)
        predictions_by_tag = dict()
        for l in range(len(out_test)):
            tag = tags[l]
            original_tag = tag
            if original_tag in predictions_by_tag:
                predictions_by_tag[original_tag].append(out_test[l].item())
            else:
                predictions_by_tag[original_tag] = [out_test[l].item()]
        sorted_prediction_by_tag = sorted(list(predictions_by_tag.keys()))
        print('marker', 'sample', 'prediction', 'standard_deviation', '25', '50', '75', sep='\t')
        df_rows = []
        for tag in sorted_prediction_by_tag:
            predictions = predictions_by_tag[tag]
            predictions.sort()
            mean_prediction = np.mean(predictions)
            std = np.around(np.std(predictions), decimals=2)
            percentile_25 = np.percentile(predictions, 25)
            percentile_50 = np.percentile(predictions, 50)
            percentile_75 = np.percentile(predictions, 75)
            print(marker, tag, "%.1f" % mean_prediction, std, percentile_25, percentile_50, percentile_75,
                sep='\t')
            df_rows.append([tag, np.round(mean_prediction, 2), std, percentile_25, percentile_50, percentile_75])
            return np.round(mean_prediction, 2), std, percentile_25, percentile_50, percentile_75
        print('\n')
        df = pd.DataFrame(df_rows, columns=['sample_name', 'prediction', 'standard_deviation', '25', '50', '75'])

        # df = md.merge(df, on='sample_name', how='left')
        # df.sort_values(by='sample_name', inplace=True)
        # df.to_csv(output_dir + "/_results_" + marker + ".csv")

