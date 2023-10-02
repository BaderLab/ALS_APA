import os, random
from unicodedata import name
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_1 import apaDNNModel, apaDataset
import pickle
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
import seqlogo


def main(device='cuda'):

    def plot_filter_logos(filters):
        for i in range(10,len(filters)):
            print(np.array(filters[i]))
            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            cpm = seqlogo.CompletePm(np.array(filters[i]), alphabet_type='RNA')
            print(cpm)
            # outname = "logs/" + cpm.consensus + "{}_.png".format(i)
            # # seqlogo.seqlogo the filter for each ax[i]
            # seqlogo.seqlogo(cpm, ax=ax[i], ic_scale=False, format='png', size='medium')
            # # save the figure
            # plt.savefig(outname)
            break
    
    def test_performance(model, test_data_loader, device):
        target_list, pred_list = [], []
        valid_loss, valid_R = 0.0, 0.0
        cells_res = {}
        model.eval()
        # make a empty torch tensor to save the filters of the first conv layer
        filters = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                seq_X, celltype, celltype_name, Y = batch
                out = model(seq_X, celltype)
                # filters:
                # get the filters of the first conv layer
                filters.append(model.conv_block_1.op[0].weight.data)
            stacked_activations = torch.stack(filters, dim=0).to(device)
            print("stacked_activations shape: ", stacked_activations.shape)
            mean_activations = torch.mean(stacked_activations, dim=0).to(device)
            print("mean_activations shape: ", mean_activations.shape)
            #
            ## lets turn the activatoions into RNA pwms
            ppms = F.softmax(mean_activations, dim=1)
            # loop over the filters and save in a list:
            filter_lst = []
            for f in range(ppms.shape[0]):
                filter_lst.append(ppms[f,:,:].cpu().numpy())
            # plot the filters:
            plot_filter_logos(filter_lst)

                

            

    data_root = "/data/users/goodarzilab/aiden/projects/APA/input_data/V2/"
    test_data = np.load(data_root + "C9ALS_test_labels.npy", allow_pickle=True)
    test_seq = np.load(data_root + "C9ALS_test_seqs.npy", allow_pickle=True)

    profiles = pd.read_csv(data_root + "celltype_profiles.tsv", index_col=0, sep="\t")
    test_data_loader = DataLoader(
        apaDataset(test_seq, test_data, profiles, device='cuda:2'),
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )

    model = apaDNNModel(
        opt="Adam",
        loss="mse",
        lambda1=50,
        device=device,
        # Conv block 1 hparamaters
        conv1kc=128,
        conv1ks=12,
        conv1st=1,
        pool1ks=20,
        pool1st=20,
        cnvpdrop1=0.2,
        # multihead attention block
        Matt_heads=8,
        Matt_drop = 0.2,
        # FC block 1 (Matt output flattened)
        fc1_L1=0,  # 8192
        fc1_L2=8192,
        fc1_L3=4048,
        fc1_L4=1024,
        fc1_L5=512,
        fc1_L6=256,
        fc1_dp1=0.3,
        fc1_dp2=0.25,
        fc1_dp3=0.25,
        fc1_dp4=0.2,
        fc1_dp5=0.1,
        # FC block 2 (celltype profile + overall representation)
        fc2_L1=0,
        fc2_L2=128,
        fc2_L3=32,
        fc2_L4=16,
        fc2_L5=1,
        fc2_dp1=0.2,
        fc2_dp2=0.2,
        fc2_dp3=0,
        fc2_dp4=0,
        lr=2.5e-05,
        adam_weight_decay=0.06,
    )
    model.compile()
    model.load_state_dict(
        torch.load(
            "/data/users/goodarzilab/aiden/projects/APA/input_data/model_outs/all_cells_CNN_Matt_V2_4_L2reg_bestModel.pt"
        )
    )
    model.to("cuda:2")
    test_performance(model, test_data_loader, device = 'cuda:2')


if __name__ == "__main__":
    ## setting the random seeds to a fixed number for maximum reproducibilty
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    main(device="cuda:2")
