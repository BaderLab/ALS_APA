import os, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock, FC_block


class apaDataset(Dataset):
    """
    apaDataset is the data seq_itrator for the apaDNN model
    Arguments
    ---------
    df:    Pandas dataframe contating the sampels
    max_len:  maximum length of the sequences
    lef_pad_max: maximum allowed padding on the left side of the sequences
    """

    def __init__(self, seqs, df, ct, device):
        self.device = device
        self.reg_label = torch.from_numpy(
            np.array(df[:, 3].tolist(), dtype=np.float32)
        ).to(device)
        # self.class_label = torch.from_numpy(
        #     np.array(df[:, 5].tolist(), dtype=np.float32)
        # ).to(device)
        self.celltypes = df[:, 2]

        self.seq_idx = torch.from_numpy(np.array(df[:, 1].tolist(), dtype=np.int32)).to(
            device
        )

        self.oneH_seqs = torch.from_numpy(np.array(list(seqs[:, 3]), dtype=np.int8)).to(
            device
        )
        self.oneH_seq_indexes = torch.from_numpy(
            np.array(seqs[:, 0], dtype=np.int32)
        ).to(device)

        self.ct_profiles = ct

    def __len__(self):
        return self.reg_label.shape[0]

    def __getitem__(self, idx):
        seq_idx = self.seq_idx[idx]
        seq = (
            self.oneH_seqs[torch.where(self.oneH_seq_indexes == seq_idx)]
            .squeeze()
            .type(torch.cuda.FloatTensor)
        )
        reg_label = self.reg_label[idx]
        # class_label = self.class_label[idx]
        celltype_name = self.celltypes[idx]
        celltype = torch.from_numpy(
            self.ct_profiles[celltype_name].values.astype(np.float32)
        ).to(self.device)
        return (seq, celltype, celltype_name, reg_label)


class apaDNNModel(nn.Module):
    def __init__(
        self,
        opt="Adam",
        loss="mse",
        lambda1=0.01,
        lr=3e-4,
        device="cuda",
        adam_weight_decay=0.07,
        # Conv block 1 hparamaters
        conv1kc=128,
        conv1ks=12,
        conv1st=1,
        pool1ks=20,
        pool1st=20,
        cnvpdrop1=0,
        # multihead attention block
        Matt_heads=1,
        Matt_drop=0,
        # FC block 1 (Matt output flattened)
        fc1_L1=0,  # 8192
        fc1_L2=8192,
        fc1_L3=4048,
        fc1_L4=1024,
        fc1_L5=512,
        fc1_L6=256,
        fc1_dp1=0,
        fc1_dp2=0,
        fc1_dp3=0,
        fc1_dp4=0,
        fc1_dp5=0,
        # FC block 2 (celltype profile + overall representation)
        fc2_L1=0,
        fc2_L2=128,
        fc2_L3=32,
        fc2_L4=16,
        fc2_L5=1,
        fc2_dp1=0,
        fc2_dp2=0,
        fc2_dp3=0,
        fc2_dp4=0,
    ):
        super(apaDNNModel, self).__init__()
        self.opt = opt
        self.loss = loss
        self.lr = lr
        self.device = device
        self.adam_wd = adam_weight_decay
        
        def get_conv1d_out_length(l_in, padding, dilation, kernel, stride):
            """
            Returns the length of sequence after convolution steps.
            Formula adapted from pytorch documentation
            """
            return int((l_in + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1

        self.conv_block_1 = ConvBlock(
            4,
            conv1kc,
            cnvks=conv1ks,
            cnvst=conv1st,
            poolks=pool1ks,
            poolst=pool1st,
            pdropout=cnvpdrop1,
            activation_t="ELU",
        )
        cnv1_len = get_conv1d_out_length(4000, conv1ks // 2, 1, conv1ks, conv1st)
        cnv1_len = get_conv1d_out_length(cnv1_len, 0, 1, pool1ks, pool1st)

        # Attention block
        self.Matt_1 = nn.MultiheadAttention(embed_dim=conv1kc, num_heads=Matt_heads, dropout=Matt_drop)
        # FC block 1
        fc1_L1 = cnv1_len * conv1kc
        self.fc1 = FC_block(
            [fc1_L1, fc1_L2, fc1_L3, fc1_L4, fc1_L5, fc1_L6],
            [fc1_dp1, fc1_dp2, fc1_dp3, fc1_dp4, fc1_dp5],
            dropout=True,
        )

        # FC block 2
        fc2_L1 = fc1_L6 + 279
        self.fc2 = FC_block(
            [fc2_L1, fc2_L2, fc2_L3, fc2_L4, fc2_L5],
            [fc2_dp1, fc2_dp2, fc2_dp3, fc2_dp4],
            dropout=True,
        )

    def forward(self, seq, celltype):

        x_conv = self.conv_block_1(seq)
        # reshape CNN output to fit the attention block (sequence, batch, embedding size)
        x = x_conv.permute(2, 0, 1)
        x, w = self.Matt_1(x, x, x, need_weights=True)
        x = x.permute(1, 2, 0)  # back to batch, embedding size, sequence
        # add the residual connection
        x = x + x_conv
        # flatten
        x = torch.flatten(x, 1)
        # first FC on the sequence representation
        x = self.fc1(x)
        # concat the final representation with celltype profile
        x = torch.cat((x, celltype), 1)
        # second FC on the concatenated representations
        x = self.fc2(x)
        return x

    def compile(self):
        device = self.device
        self.to(device)
        if self.opt == "Adam":
            self.optimizer = optim.AdamW(
                self.parameters(),
                weight_decay=self.adam_wd,
                amsgrad=False,
                lr=self.lr,
            )  # check the paper for amsgrad
        if self.loss == "mse":
            self.loss_fn = nn.MSELoss()

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
