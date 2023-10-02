from captum.attr import DeepLift
import os, random
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_1 import apaDNNModel, apaDataset
import pickle
import seaborn as sns



SHUFFLE_NUM = 10


## some functions are adopted from Kundaje Lab Github rep and modified for our purpose.


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an D x L one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    print(one_hot.shape[1])
    print(one_hot.shape[0])
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    print(tokens.shape)
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        one_hot_dim, seq_len = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an D x L one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[0], one_hot.shape[1])  # Vector of all D
    dim_inds, seq_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return np.transpose(identity[tokens])


def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        one_hot_dim, seq_len = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, one_hot_dim, seq_len), dtype=seq.dtype
        )
    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def combine_mult_and_diffref(mult, orig_inp, ref):
    """
    This is a custom function to calculate the hypothetical contribution
    of bases of a one hot sequence for a deeplearing model. 
    inputs:
    mult: multipliers from deeplift
    orig_inp: the original input or tuple of inputs to the model
    ref: the base or the tuple of bases for the deeplift model
    returns the tuple including the hypothetical contributon of bases/other inputs 
    """
    to_return = []
    mult = [x.detach().cpu().numpy() for x in mult]
    orig_inp = [x.detach().cpu().numpy() for x in orig_inp]
    ref = [x.detach().cpu().numpy() for x in ref]

    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(ref[l]).astype("float")
        # assert len(orig_inp[l].shape)==2, orig_inp[l].shape
        if (
            len(orig_inp[l].shape) == 3
        ):  # for case of batch x 1h_dim x seq_len (minibatch x 4 x 16)
            for i in range(
                orig_inp[l].shape[1]
            ):  # iterating over 1h dim and changing the bases to other base
                hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
                hypothetical_input[:, i, :] = 1.0
                difference_from_reference = hypothetical_input - ref[l]
                hypothetical_contribs = difference_from_reference * mult[l]
                projected_hypothetical_contribs[:, i, :] = np.mean(
                    hypothetical_contribs, axis=1
                )
            to_return.append(projected_hypothetical_contribs)
        else:  ## for the case of celltypes # MINI_BATCH x 14 >> and we don't care about this one
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            difference_from_reference = hypothetical_input - ref[l]
            hypothetical_contribs = difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, i] = np.mean(
                hypothetical_contribs, axis=-1
            )
            to_return.append(projected_hypothetical_contribs)
    return tuple(to_return)


def get_batch_attr(DL_model, seqs, celltype, celltype_base, shuffled_seqs, device, SHUFFLE_NUM):
    """
    calculates the attributions of sequence bases based on 10 different dinucleotide shuffled sequences
    for both the main sequence and also hypothetical contributions of other bases in the sequence
    returns a tuple of input sequence base attributions and hypothetical contributions of other bases
    """
    shuff_attr = []
    hypo_contr = []
    for i in range(SHUFFLE_NUM):
        base_seqs = torch.from_numpy(
            np.array(
                [shuffled_seqs[j][i] for j in range(seqs.shape[0])], dtype=np.float32
            )
        )
        inp = (
            seqs.to(device, non_blocking=True),
            celltype.to(device, non_blocking=True),
        )
        base = (base_seqs.to(device, non_blocking=True), celltype_base)
        seq_attr, _ = DL_model.attribute(inp, baselines=base)
        hypo_attr, _ = DL_model.attribute(
            inp, baselines=base, custom_attribution_func=combine_mult_and_diffref
        )
        shuff_attr.append(seq_attr.detach().to("cpu").numpy())
        hypo_contr.append(hypo_attr)
    inp_seq_attr = np.array(shuff_attr, dtype=np.float32)
    hypothetical_attr = np.array(hypo_contr, dtype=np.float32)

    return (
        np.mean(inp_seq_attr, axis=0, dtype=np.float32),
        np.mean(hypothetical_attr, axis=0, dtype=np.float32),
    )


if __name__ == "__main__":
    device = 'cuda:2'
    data_root = "/data/users/goodarzilab/aiden/projects/APA/input_data/V2/"
    profiles = pd.read_csv(data_root + "celltype_profiles.tsv", index_col=0, sep="\t")
    test_data = np.load(data_root + "C9ALS_test_labels.npy", allow_pickle=True)
    test_seq = np.load(data_root + "C9ALS_test_seqs.npy", allow_pickle=True)

    MINI_BATCH = 128
    df_data_loader = DataLoader(apaDataset(test_seq, test_data, profiles, device),
        batch_size=MINI_BATCH,
        shuffle=False,
        drop_last=False)
    SAMPLE_SIZE = test_data.shape[0]
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
    DL_model = DeepLift(model)
    all_attr, hyp_attr = (
        np.zeros((test_data.shape[0], 4, 4000)),
        np.zeros((test_data.shape[0], 4, 4000)),
    )
    for batch_idx, batch in enumerate(df_data_loader):
        torch.cuda.empty_cache()
        print("calculating the attributions for mini-batch number: ", batch_idx)
        start_idx = batch_idx * MINI_BATCH
        end_idx = min(start_idx + MINI_BATCH, SAMPLE_SIZE)
        # seqid, seq_X, y, celltype = batch
        seq_X, celltype, cn, Y = batch
        celltype_base = torch.zeros(celltype.shape).to(device)
        batch_shuffled_seqs = []
        for i in range(seq_X.shape[0]):
            batch_shuffled_seqs.append(
                dinuc_shuffle(seq_X[i].to("cpu").numpy(), SHUFFLE_NUM)
            )
        all_attr[start_idx:end_idx], hyp_attr[start_idx:end_idx] = get_batch_attr(
            DL_model,
            seq_X,
            celltype,
            celltype_base,
            batch_shuffled_seqs,
            device,
            SHUFFLE_NUM,
        )
    outname = (
        "/data/users/goodarzilab/aiden/projects/APA/input_data/model_interpretations/"
        + "all_cells_DL"
        + "_attr.npz"
    )
    np.savez_compressed(
        outname, base_attr=all_attr, hypo_contr=hyp_attr,
    )