# -*- coding: utf-8 -*-
import sys
import numpy as np
from sklearn.datasets import fetch_mldata

import renom as rm
from renom_tda.topology import Topology
from renom_tda.lens import PCA, TSNE
from renom_tda.lens_renom import AutoEncoder

class MTae(rm.Model):
    def __init__(self):
        self.conv1 = rm.Dense(10)
        self.conv2 = rm.Dense(2)
        self.deconv1 = rm.Dense(10)
        self.deconv2 = rm.Dense(784)
        
    def forward(self, x):
        h = self.encode(x)
        h = self.deconv1(h)
        h = rm.relu(h)
        y = self.deconv2(h)
        loss = rm.mse(y, x)
        return loss
    
    def encode(self, x):
        h = self.conv1(x)
        h = rm.relu(h)
        h = self.conv2(h)
        return h
    
def main(mode):
    # metric & lens    
    metric = None
    if mode == 'pca':
        lens=[PCA(components=[0,1])]
    elif mode == 'tsne':
        lens=[TSNE(components=[0,1])]
    elif mode == 'ae':
        lens = [AutoEncoder(epoch=100,
                            batch_size=256,
                            network=MTae(),
                            opt=rm.optimizer.Adam(),
                            verbose=1)]
    else:
        raise Exception
        
    # dataset
    mnist = fetch_mldata('MNIST original', data_home="dataset")
    # subtraction, 1/10
    data = mnist.data[::10]
    target = mnist.target[::10]
    data = data.astype(np.float32)/255.0
    
    # topology
    topology = Topology()
    topology.load_data(data)
            
    topology.fit_transform(metric=metric, lens=lens)
    
    topology.map(resolution=50, overlap=1, eps=0.4, min_samples=2)
    topology.color(target, color_method='mode', color_type='rgb')
    topology.show(fig_size=(15,15), node_size=1, edge_width=0.1, mode='spring', strength=0.05)
    
if __name__ == '__main__':
    args = sys.argv
    try:
        if len(args) < 2:
            raise Exception
        else:
            mode = args[1]
            main(mode)
    except Exception:
        print('ERROR: python MTRenomtda.py \'mode(pca or tsne or ae)\'')
