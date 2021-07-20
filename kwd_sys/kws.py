import numpy as np 
import torch
import kaldiio
from scipy.spatial.distance import cdist
# import os; os.environ["CUDA_VISIBLE_DEVICES"]="2"

from Models import *

# config_file = sys.argv[1]

class MyKWS:
    def __init__(self, model_path, use_gpu=True):
        self.model_name = 'ConvNet'
        self.max_height = 100
        self.max_width  = 300
        self.use_gpu    = torch.cuda.is_available()
        self.model      = self.load_saved_model(model_path)
        self.model.eval()

    # Function to load saved models for evaluation on test data
    # Expected input is a config dict with the model name (ConvNet, VGG, ResNet34) to paths(s)
    def load_saved_model(self, model_path):

        constructor = globals()[self.model_name]
        model = constructor(self.max_height, self.max_width)

        if self.use_gpu: model.cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def predict(self, query_ark, test_ark):
        query = kaldiio.load_mat(query_ark)
        test = kaldiio.load_mat(test_ark)
        qt_dists    = cdist(query, test, 'seuclidean', V = None)
        qt_dists    = -1 + 2 * ((qt_dists - qt_dists.min())/(qt_dists.max() - qt_dists.min()))

        # Get indices to downsample or pad M x N matrix to max_height x max_width (default 100 x 800)
        def get_keep_indices(dim_size, dim_max):
            if dim_size <= dim_max:
                # no need to downsample if M or N smaller than max_height/max_width
                return np.arange(0, dim_size)
            else:
                # if bigger, return evenly spaced indices for correct height/width
                return np.round(np.linspace(0, dim_size - 1, dim_max)).astype(int)

        ind_rows = get_keep_indices(qt_dists.shape[0], self.max_height)
        ind_cols = get_keep_indices(qt_dists.shape[1], self.max_width)

        qt_dists = np.take(qt_dists, ind_rows, axis = 0)
        qt_dists = np.take(qt_dists, ind_cols, axis = 1)

        # Create empty 100 x 800 matrix, then fill relevant cells with dist values
        temp_dists = np.full((self.max_height, self.max_width), qt_dists.min(), dtype='float32')
        temp_dists[:qt_dists.shape[0], :qt_dists.shape[1]] = qt_dists

        # Reshape to (1xHxW) since to feed into ConvNet with 1 input channel
        dists = torch.Tensor(temp_dists).view(1, 1, self.max_height, self.max_width)

        # Move data to GPU if desired
        if self.use_gpu:
            dists = dists.cuda()

        outputs = self.model(dists)

        outputs = outputs.cpu().detach().numpy().round(10)

        return outputs[0][0]

# if '__name__' == '__main__':
#     model_path = 'models/model_autoKWS2021_ConvNet_noaug_100x100.pt'
#     use_gpu = False
#     query_ark = '../../temp_output/P0001/bnf/enroll/raw_bnfeat_enroll.1.ark:17'
#     test_ark  = '../../temp_output/P0001/bnf/test/raw_bnfeat_test.5.ark:65064'

#     my_kws = MyKWS(model_path, use_gpu)
#     result = my_kws.predict(query_ark, test_ark)
#     print(result)