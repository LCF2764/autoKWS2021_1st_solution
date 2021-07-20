import os
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import kaldiio
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
import logging
import matplotlib.pyplot as plt 

class STD_Dataset(Dataset):
    """Spoken Term Detection dataset."""

    def __init__(self, root_dir, labels_csv, feats_scp, apply_vad = False, max_height=100, max_width=300):
        """
        Args:
            root_dir (string): Absolute path to dataset directory with content below
            labels_csv (string): Relative path to the csv file with query and test pairs, and labels
                (1 = query in test; 0 = query not in test).
            query_dir (string): Relative path to directory with all the audio queries.
            audio_dir (string): Relative path to directory with all the test audio.
        """

        if isinstance(labels_csv, dict):
            # Supplying separate csv files for positive and negative labels
            pos_frame   = pd.read_csv(os.path.join(root_dir, labels_csv['positive_labels']))
            neg_frame   = pd.read_csv(os.path.join(root_dir, labels_csv['negative_labels']))
            # Randomly down-sample neg examples to same number of positive examples
            pos_frame   = pos_frame.sample(frac = labels_csv['pos_sample_size'], replace = True)
            neg_frame   = neg_frame.sample(n = pos_frame.shape[0])

            self.qtl_frame = pd.concat([pos_frame, neg_frame], axis = 0).sample(frac = 1)
        else:
            # If a single CSV file, then just read that in
            self.qtl_frame  = pd.read_csv(os.path.join(root_dir, labels_csv))

        self.apply_vad  = apply_vad
        self.max_height = max_height
        self.max_width  = max_width

        # read bnf-scp to dict
        self.bnfeats_scp_dict = {}
        for line in open(os.path.join(root_dir, feats_scp), 'r').readlines():
            utt, ark = line.strip().split()
            self.bnfeats_scp_dict[utt] = ark

        # if apply_vad is True:
        #     # If using voice activity detection we expect same directory structure
        #     # and file names as feature files for .npy files containing voice activity
        #     # detection (VAD) labels (0 = no speech activity, 1 = speech activity)
        #     # in a 'vad_labels' directory
        #     self.vad_query_dir = os.path.join(root_dir, 'vad_labels', query_dir)
        #     self.vad_audio_dir = os.path.join(root_dir, 'vad_labels', audio_dir)

        #     # Get filenames in audio and query directories
        #     q_files = os.listdir(self.vad_query_dir)
        #     a_files = os.listdir(self.vad_audio_dir)

        #     # Get length of non-zero values in files
        #     q_vlens = np.array([ len(np.flatnonzero(np.load(os.path.join(self.vad_query_dir, f)))) for f in q_files ])
        #     a_vlens = np.array([ len(np.flatnonzero(np.load(os.path.join(self.vad_audio_dir, f)))) for f in a_files ])

        #     # Get files (without .npy extensions) for which there are no non-zero values
        #     zero_qs = [ os.path.splitext(x)[0] for x in np.take(q_files, np.where(q_vlens == 0)).flatten() ]
        #     zero_as = [ os.path.splitext(x)[0] for x in np.take(a_files, np.where(a_vlens == 0)).flatten() ]

        #     if(len(zero_qs) > 0):
        #         logging.info(" Following queries removed from dataset (insufficient frames after VAD): %s" % (", ".join(zero_qs)))

        #     if(len(zero_as) > 0):
        #         logging.info(" Following references removed from dataset (insufficient frames after VAD): %s" % (", ".join(zero_as)))

        #     # Discard from labels irrelevant files
        #     self.qtl_frame = self.qtl_frame[~self.qtl_frame['query'].isin(zero_qs)]
        #     self.qtl_frame = self.qtl_frame[~self.qtl_frame['reference'].isin(zero_as)]

    def __len__(self):
        return len(self.qtl_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        query_name = self.qtl_frame.iloc[idx, 0] 
        test_name  = self.qtl_frame.iloc[idx, 1] 
        qt_label   = self.qtl_frame.iloc[idx, 2]

        # Get features where query = M x f, test = N x f, where M, N number of frames and f number of features
        query_feat = kaldiio.load_mat(self.bnfeats_scp_dict[query_name])
        test_feat = kaldiio.load_mat(self.bnfeats_scp_dict[test_name])

        # if self.apply_vad is True:
        #     query_vads = np.load(os.path.join(self.vad_query_dir, query_name + ".npy"), allow_pickle=True)
        #     test_vads  = np.load(os.path.join(self.vad_audio_dir, test_name + ".npy"), allow_pickle=True)

        #     # Keep only frames (rows, axis = 0) where voice activity detection by rVAD has returned non-zero (i.e. 1)
        #     query_feat = np.take(query_feat, np.flatnonzero(query_vads), axis = 0)
        #     test_feat  = np.take(test_feat, np.flatnonzero(test_vads), axis = 0)

        # Create standardised Euclidean distance matrix of dimensions M x N
        qt_dists    = cdist(query_feat, test_feat, 'seuclidean', V = None)
        # Range normalise matrix to [-1, 1]
        qt_dists    = -1 + 2 * ((qt_dists - qt_dists.min())/(qt_dists.max() - qt_dists.min()))

        # Get indices to downsample or pad M x N matrix to max_height x max_width (default 100 x 800)
        def get_keep_indices(dim_size, dim_max):
            if dim_size <= dim_max:
                # no need to downsample if M or N smaller than max_height/max_width
                return np.arange(0, dim_size)
            else:
                # if bigger, return evenly spaced indices for correct height/width
                return np.round(np.linspace(0, dim_size - 1, dim_max)).astype(int)

        Resize = transforms.Compose([transforms.Resize(size=(self.max_height, int(self.max_height*qt_dists.shape[1]/qt_dists.shape[0])))])
        qt_dists = torch.from_numpy(qt_dists).unsqueeze(0)
        qt_dists = Resize(qt_dists).squeeze().numpy()
        # ind_rows = get_keep_indices(qt_dists.shape[0], self.max_height)
        ind_cols = get_keep_indices(qt_dists.shape[1], self.max_width)

        # qt_dists = np.take(qt_dists, ind_rows, axis = 0)
        qt_dists = np.take(qt_dists, ind_cols, axis = 1)

        # Create empty 100 x 800 matrix, then fill relevant cells with dist values
        temp_dists = np.full((self.max_height, self.max_width), qt_dists.min(), dtype='float32')
        temp_dists[:qt_dists.shape[0], :qt_dists.shape[1]] = qt_dists

        # Reshape to (1xHxW) since to feed into ConvNet with 1 input channel
        dists = torch.Tensor(temp_dists).view(1, self.max_height, self.max_width)
        label = torch.Tensor([qt_label])

        sample = {'query': query_name, 'reference': test_name, 'dists': dists, 'labels': label}

        return sample
