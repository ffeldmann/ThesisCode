import os
import numpy as np
import skimage
from edflow.data.believers.meta import MetaDataset
from edflow.data.believers.meta_view import MetaViewDataset

class SingleCats(MetaDataset):
    def __init__(self, config):
        #self.config = config
        super().__init__(config["dataroot"])
        #self.append_labels = True
        #self.expand = True

    #def __len__(self):
    #    return min(len(super()), self.config["max_dataset_size"])