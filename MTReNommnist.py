# -*- coding: utf-8 -*-
import argparse
import numpy as np
import renom as rm

from sklearn.metrics import confusion_matrix, classification_report

def main():
    parser = argparse.ArgumentParser(description='ReNom example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='use GPU (negative value indicates CPU)')
    parser.add_argument('--train', '-t', type=int, default=1,
                        help='If negative, skip training')
    parser.add_argument('--resume', '-r', type=int, default=-1,
                        help='If positive, resume the training from snapshot')
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    flag_train = False if args.train < 0 else True
    flag_resum = False if args.resume < 0 else True
    n_epoch = args.epoch if flag_train == True else 1
    
    tsm = MTRNClass(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize)
    tsm.run()
    
class MTRNClass():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize):
        rm.core.DEBUG_GRAPH_INIT(True)
        if not gpu < 0:
            rm.cuda.cuda.set_cuda_active(True)
        
        self.flag_train = flag_train
        self.n_epoch = n_epoch
        self.bs = batchsize
        
        self.model = rm.Sequential([
                rm.Dense(256),
                rm.Relu(),
                rm.Dropout(dropout_ratio=0.5),
                rm.Dense(10)
                ])
        
        if flag_resum:
            try:
                self.model.load('model.h5')
            except:
                print('ERROR: can not resume pre-trained model')
            
        self.opt = rm.optimizer.Adam()
        self.prepare_mnist()
        
    def run(self):
        for epoch in range(self.n_epoch):
            perm = np.random.permutation(self.N_train)
            total_loss = 0
            for j in range(self.epoch_loop):
                x, t = self.next_batch(perm,j)
                with self.model.train():
                    y = self.model(x)
                    loss = rm.softmax_cross_entropy(y, t)
                    
                loss.to_cpu()
                grad = loss.grad()
                grad.update(self.opt)
                total_loss += loss
                
            train_loss = loss / (self.N_train // self.bs)
            train_loss.to_cpu()
            test_loss = rm.softmax_cross_entropy(self.model(self.X_test), self.labels_test)
            test_loss.to_cpu()
            print('epoch {} train_loss:{} test_loss:{}'.format(epoch, train_loss, test_loss))
        
        ret = self.model(self.X_test)
        ret.to_cpu()
        predictions = np.argmax(np.array(ret), axis=1)
        
        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))
        rm.core.DEBUG_NODE_STAT()
        
        if self.flag_train:
            self.model.save('model.h5')
    
    def prepare_mnist(self):
        # prepare dataset
        from sklearn.datasets import fetch_mldata
        from sklearn.cross_validation import train_test_split
        from sklearn.preprocessing import LabelBinarizer
        
        mnist = fetch_mldata('MNIST original', data_home="dataset")
        X = mnist.data
        y = mnist.target
        X = X.astype(np.float32)/255.0
        
        self.X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.1)
        self.labels_train = LabelBinarizer().fit_transform(y_train).astype(np.float32)
        self.labels_test = LabelBinarizer().fit_transform(self.y_test).astype(np.float32)
        
        self.N_train = len(self.X_train)
        self.epoch_loop = self.N_train // self.bs
        
    def next_batch(self, perm, j):
        x = self.X_train[perm[j*self.bs:(j+1)*self.bs]]
        t = self.labels_train[perm[j*self.bs:(j+1)*self.bs]]
        return x, t
        
if __name__ == '__main__':
    main()
