
import os, random
from unicodedata import name
import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
import seaborn as sns
import modisco
import modisco.visualization
from modisco.visualization import viz_sequence
import h5py


data_root = "/data/users/goodarzilab/aiden/projects/APA/input_data/V2/"
attr_root = '/data/users/goodarzilab/aiden/projects/APA/input_data/model_interpretations/all_cells_DL_attr.npz'
test_seq = np.load(data_root + "C9ALS_test_seqs.npy", allow_pickle=True)
test_data = np.load(data_root + "C9ALS_test_labels.npy", allow_pickle=True)

profiles = pd.read_csv(data_root + "celltype_profiles.tsv", index_col=0, sep="\t")

att_hat_load = np.load(attr_root, allow_pickle=True)
seq_att, hyp_att = [value for key, value in att_hat_load.items()]
seq_dict = dict(zip(test_seq[:, 1], test_seq[:, 3]))
seq_ids = test_data[:, 0]
# lets get the onehotar for all the seq_ids, a list with all the 52669 seqs
onehotar = []
for id in seq_ids:
    onehotar.append(seq_dict[id].T)

# mean normalizing the hypothetical contributions
hyp_att = hyp_att - np.mean(hyp_att, axis=1, keepdims=True)

dl_dict = {"task0": [each.T for each in seq_att]}
dl_dict_hyp_cont = {"task0": [each.T for each in hyp_att]}
null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
    # Slight modifications from the default settings >> try 7 and 10
    sliding_window_size=12,
    flank_size=5,
    target_seqlet_fdr=0.15,
    seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
        trim_to_window_size=12,
        initial_flank_to_add=3,
        final_flank_to_add=3,
        final_min_cluster_size=60,
        n_cores=10,
    ),
)(
    task_names=["task0"],
    contrib_scores=dl_dict,
    hypothetical_contribs=dl_dict_hyp_cont,
    one_hot=onehotar,
    null_per_pos_scores=null_per_pos_scores,
    revcomp=False,
)

grp = h5py.File("/data/users/goodarzilab/aiden/projects/APA/input_data/model_interpretations/TF_modisco_All_CTs_results.hdf5", "w")
tfmodisco_results.save_hdf5(grp)
grp.close()
