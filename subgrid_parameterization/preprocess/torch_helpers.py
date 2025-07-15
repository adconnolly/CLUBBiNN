import numpy as np
import torch.utils.data as Data # The ready-to-use class for dumbies, e.g. Alex Connolly, to write this code

def split_dataset(dataset,ratio):
        
        x,y=dataset.tensors
        
        n=len(x)
        train_n=int(n*ratio)
        
        rand_indices=np.random.permutation(np.arange(n))
        train_indices = rand_indices[:train_n]
        test_indices = rand_indices[train_n:]
    
        x_train=x[train_indices]
        x_test=x[test_indices]
        y_train=y[train_indices]
        y_test=y[test_indices]
    
        return Data.TensorDataset(x_train,y_train),Data.TensorDataset(x_test,y_test)