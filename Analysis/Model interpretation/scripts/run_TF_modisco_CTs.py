
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
attr_root = '/data/users/goodarzilab/aiden/projects/APA/input_data/model_interpretations/'
test_seq = np.load(data_root + "C9ALS_test_seqs.npy", allow_pickle=True)
test_data = np.load(data_root + "C9ALS_test_labels.npy", allow_pickle=True)

profiles = pd.read_csv(data_root + "celltype_profiles.tsv", index_col=0, sep="\t")

for ct in profiles.columns:
    try:
        print(ct)
        # load the hypothetical contributions and the sequence attributions
        att_file = attr_root + '{}_attr.npz'.format(ct)
        ct_hat_load = np.load(att_file, allow_pickle=True)
        ct_seq_att, ct_hyp_att = [value for key, value in ct_hat_load.items()]
        test_data_ct = test_data[test_data[:, 2] == ct]
        seq_ids = test_data_ct[:, 0]
        # filter the test_seq based on seq_ids, seq_ids are in teh second column of test_seq
        onehotar = [each[3].T for each in test_seq if each[1] in seq_ids]

        # mean normalizing the hypothetical contributions
        ct_hyp_att = ct_hyp_att - np.mean(ct_hyp_att, axis=1, keepdims=True)

        dl_dict = {"task0": [each.T for each in ct_seq_att]}
        dl_dict_hyp_cont = {"task0": [each.T for each in ct_hyp_att]}

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

        grp = h5py.File("/data/users/goodarzilab/aiden/projects/APA/input_data/model_interpretations/TF_modisco_results_{}.hdf5".format(ct), "w")
        tfmodisco_results.save_hdf5(grp)
        grp.close()
    except:
        continue
