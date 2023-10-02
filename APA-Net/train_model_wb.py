import os, random
from unicodedata import name
import pandas as pd
import numpy as np
import scipy.stats as stats
from torch.utils.data import Dataset, DataLoader
from model_1 import apaDNNModel, apaDataset
import pickle
import wandb
import torch

wandb.login()


def build_dataset(
    device, train_seq, valid_seq, train_data, val_data, batch_size, ct_profiles
):
    """
    takes in the batch size and then generates validation and
    training dataloaders.
    """

    # ## read datasets to the GPU

    train_data_loader = DataLoader(
        apaDataset(train_seq, train_data, ct_profiles, device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_data_loader = DataLoader(
        apaDataset(valid_seq, val_data, ct_profiles, device),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_data_loader, valid_data_loader


def train_set_train(model, train_loader):
    """
    training on the whole training dataset for one epoch
    we have prefetch mechanism here so we can put the
    data batch in gpu while training the previous batch
    the function returns training loss and correlation coefficient
    per one epoch
    """
    device = model.device
    model.train()
    train_pred_list, train_target_list = [], []
    train_loss_acc = 0
    for batch_idx, batch in enumerate(train_loader):
        model.optimizer.zero_grad()
        seq_X, celltype, cellname, Y = batch
        ## forward pass, backprob, update
        out = model(seq_X, celltype)
        out = torch.squeeze(out)
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.norm(param)
        loss = torch.sqrt(model.loss_fn(out, Y))
        train_pred_list.append(out.to("cpu").detach().numpy())
        train_target_list.append(Y.to("cpu").detach().numpy())
        # backwards and updating the parameters
        loss.backward()
        model.optimizer.step()
        training_loss = loss.item() * seq_X.size(0)
        train_loss_acc += training_loss

    # loss per epoch
    train_loss_acc /= len(train_loader.dataset)
    # R per epoch
    train_target_list = np.concatenate(train_target_list)
    train_pred_list = np.concatenate(train_pred_list)
    train_R_epoch = stats.pearsonr(train_target_list, train_pred_list)[0]

    return train_loss_acc, train_R_epoch


def valid_set_train(model, val_loader):
    """
    training on the whole validation dataset for one epoch
    we have prefetch mechanism here so we can put the
    data batch in gpu while training the previous batch
    the function returns validation loss and correlation coefficient
    per one epoch
    """
    device = model.device
    model.eval()
    with torch.no_grad():
        target_list, pred_list = [], []
        valid_loss, valid_R = 0.0, 0.0
        for batch_idx, batch in enumerate(val_loader):
            # read in the data into gpu
            seq_X, celltype, cellname, Y = batch
            out = model(seq_X, celltype)
            out = torch.squeeze(out)
            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.norm(param)
            loss = torch.sqrt(model.loss_fn(out, Y))
            valid_loss += loss.item() * seq_X.size(0)
            pred_list.append(out.to("cpu").detach().numpy())
            target_list.append(Y.to("cpu").detach().numpy())

        targets = np.concatenate(target_list)
        preds = np.concatenate(pred_list)

        valid_R = stats.pearsonr(targets.flatten(), preds.flatten())[0]
        valid_loss /= len(val_loader.dataset)

    return valid_loss, valid_R


def train_epoch(model, train_loader, val_loader):
    """
    training on the whole training and validation
    dataset for one epoch
    """
    train_loss, train_corel = train_set_train(model, train_loader)
    valid_loss, valid_corel = valid_set_train(model, val_loader)

    return {
        "train_loss": train_loss,
        "train_corel": train_corel,
        "valid_loss": valid_loss,
        "valid_corel": valid_corel,
    }


def main_train(
    train_seq,
    valid_seq,
    train_data,
    val_data,
    profiles,
    modelfile,
    device,
    config=None,
):
    # initialize a new wandb run
    with wandb.init(
        project="all_cells_CNN_MATT_V2_4_sALS_ResNet",
        settings=wandb.Settings(start_method="thread"),
    ):
        train_loader, val_loader = build_dataset(
            device, train_seq, valid_seq, train_data, val_data, 64, profiles
        )
        ## here you should pass all the hpos you want to pass or
        #  directly make the model using config.hpo
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
        print(model)
        epochs = 200
        best_model = -1
        for epoch in range(epochs):
            print(epoch)
            out = train_epoch(model, train_loader, val_loader)
            if out["valid_corel"] > best_model:
                model.save_model(modelfile)
            wandb.log(
                {
                    "train_loss": out["train_loss"],
                    "train_corel": out["train_corel"],
                    "valid_loss": out["valid_loss"],
                    "valid_corel": out["valid_corel"],
                    "epoch": epoch,
                }
            )


if __name__ == "__main__":
    # setting the random seeds to a fixed number for maximum reproducibilty
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)

    # data on cedar
    # data_root = "/scratch/aiden/APA/input_data/all_cells/C9ALS/"
    # train_data = np.load(data_root + "train_data.npy", allow_pickle=True)
    # train_seq = np.load(data_root + "train_seq.npy", allow_pickle=True)

    # valid_data = np.load(data_root + "valid_data.npy", allow_pickle=True)
    # valid_seq = np.load(data_root + "valid_seq.npy", allow_pickle=True)

    # profiles = pd.read_csv(data_root + "all_ctrl_embeddings.csv", index_col=0,)
    # modelfile = "/scratch/aiden/APA/codes/all_cells_CNN_att_2/best_model_1.pt"

    # data on Boltzmann
    # data_root = "/home/baderlab/asababi/projects/APA/input_data/data_for_DL/all_cells/"
    # train_data = np.load(data_root + "C9ALS/train_data.npy", allow_pickle=True)
    # train_seq = np.load(data_root + "C9ALS/train_seq.npy", allow_pickle=True)

    # valid_data = np.load(data_root + "C9ALS/valid_data.npy", allow_pickle=True)
    # valid_seq = np.load(data_root + "C9ALS/valid_seq.npy", allow_pickle=True)

    # profiles = pd.read_csv(data_root + "all_ctrl_embeddings.csv", index_col=0,)
    # modelfile = (
    #     "/home/baderlab/asababi/projects/APA/codes/all_cells_apa_CNN_MATT/best_model_2.pt"
    # )

    # files on ronald
    # data_root = "/home/aiden/data/APA/input_data/data_for_DL/C9ALS/"
    # train_data = np.load(data_root + "all_celltypes/train_data.npy", allow_pickle=True)
    # train_seq = np.load(data_root + "all_celltypes/train_seq.npy", allow_pickle=True)
    # valid_data = np.load(data_root + "all_celltypes/valid_data.npy", allow_pickle=True)
    # valid_seq = np.load(data_root + "all_celltypes/valid_seq.npy", allow_pickle=True)
    # profiles = pd.read_csv(data_root + "all_ctrl_embeddings.csv", index_col=0,)
    # modelfile = (
    #     "/home/baderlab/asababi/projects/APA/codes/all_cells_apa_DiCNN/best_model.pt"
    # )

    # Assembler
    data_root = "/data/users/goodarzilab/aiden/projects/APA/input_data/V2/"
    train_data = np.load(data_root + "sALS_train_labels.npy", allow_pickle=True)
    train_seq = np.load(data_root + "sALS_train_seqs.npy", allow_pickle=True)

    valid_data = np.load(data_root + "sALS_valid_labels.npy", allow_pickle=True)
    valid_seq = np.load(data_root + "sALS_valid_seqs.npy", allow_pickle=True)

    profiles = pd.read_csv(data_root + "celltype_profiles.tsv", index_col=0, sep="\t")
    modelfile = "/data/users/goodarzilab/aiden/projects/APA/input_data/model_outs/all_cells_CNN_Matt_V2_4_L2_sALS_resNET.pt"

    main_train(
        train_seq,
        valid_seq,
        train_data,
        valid_data,
        profiles,
        modelfile,
        "cuda:1",
    )

