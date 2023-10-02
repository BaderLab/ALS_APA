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


def main(device='cuda'):
    def test_performance(model, test_data_loader, device):
        target_list, pred_list = [], []
        valid_loss, valid_R = 0.0, 0.0
        cells_res = {}
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                seq_X, celltype, celltype_name, Y = batch
                out = model(seq_X, celltype)
                out = torch.squeeze(out)
                loss = model.loss_fn(out, Y)
                valid_loss += loss.data.item() * seq_X.size(0)
                ## so this the regression model lets save the output per celltype and plot the correlation
                for i in range(len(celltype_name)):
                    if celltype_name[i] not in cells_res:
                        cells_res[celltype_name[i]] = {"pred": [], "target": []}
                    cells_res[celltype_name[i]]["pred"].append(out[i].item())
                    cells_res[celltype_name[i]]["target"].append(Y[i].item())

                # lets save the predicted and target values for the whole batch
                pred_list.append(out.to("cpu").detach().numpy())
                target_list.append(Y.to("cpu").detach().numpy())
            # loss per epoch
            valid_loss /= len(test_data_loader.dataset)
            target_list = np.concatenate(target_list)
            pred_list = np.concatenate(pred_list)
            # get correlation coefficient for the whole batch
            valid_R = stats.pearsonr(target_list, pred_list)
            R, p_val = valid_R[0], valid_R[1]
            # plot the whole batch dot plot with correlation coefficient in the plot title
            plt.figure(figsize=(10, 10))
            plt.scatter(target_list, pred_list, s=1)
            # valid_R with only two decimal points
            valid_R_str = "{:.2f}".format(R)
            valid_R_pval = "{:.2e}".format(p_val)
            plt.title("R = " + str(valid_R_str) + ' p_val = ' + str(valid_R_pval))
            plt.xlabel("Target")
            plt.ylabel("Predicted")
            plt.savefig("C9ALS_test_set_correlation.png")
            plt.close()
            # make a dataframe for the whole batch and save for future use
            df = pd.DataFrame(
                {
                    "target": target_list,
                    "pred": pred_list,
                }
            )
            df.to_csv("C9ALS_test_set_correlation.csv", index=False)
            # plot the correlation per celltype
            # also make a dataframe for each celltype and save for future use
            plt.figure(figsize=(10, 10))
            celltype_df = pd.DataFrame()
            for celltype in cells_res:
                # first lets get the correlation coefficient for each celltype
                Rstat = stats.pearsonr(
                    cells_res[celltype]["target"], cells_res[celltype]["pred"]
                )
                R, p_val = Rstat[0], Rstat[1]
                plt.scatter(
                    cells_res[celltype]["target"],
                    cells_res[celltype]["pred"],
                    s=1,
                    label=celltype,
                )
                plt.xlabel("Target")
                plt.ylabel("Predicted")
                plt.title("R = " +  "{:.2f}".format(R) + ' P_val = ' + "{:.2e}".format(p_val))
                plt.legend()
                plt.savefig("celltype_test_performances/" + celltype + "_C9ALS.png")
                plt.close()
                # make a dataframe for each celltype and save for future use
                df = pd.DataFrame(
                    {
                        "target": cells_res[celltype]["target"],
                        "pred": cells_res[celltype]["pred"],
                        'celltype': [celltype] * len(cells_res[celltype]["target"])
                    })
                # add to the celltype_df
                celltype_df = pd.concat([celltype_df, df])
                # save the celltype_df
                celltype_df.to_csv("C9ALS_test_set_correlation_celltypes.csv", index=False)
            return None

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
            pool1ks=25,
            pool1st=25,
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
            "/data/users/goodarzilab/aiden/projects/APA/input_data/model_outs/all_cells_CNN_Matt_V2_4_L2_resNET.pt"
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
