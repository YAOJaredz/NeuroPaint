import sys
sys.path.append('src')
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import random
import pickle
import os
from loader.loader_utils import DatasetDataLoader

#%%
class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, N, dt=None, tau=100, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = N
        self.tau = tau
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, N, bias=True) # input to region B
        
        # recurrent layer 
        self.h2h = nn.Linear(N, N, bias=False) 


    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        #print('init_hidden')
        #return torch.zeros(batch_size, self.hidden_size)
        return torch.randn(batch_size, self.hidden_size)*0.3

    def recurrence(self, input, hidden):
        """Run network for one time step.
        
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        #m = nn.Softplus()
        m = nn.Tanh()

        h_new = m(self.input2h(input) + self.h2h(hidden))
        hidden_new = hidden * (1 - self.alpha) + h_new * self.alpha
        
        return hidden_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden

#%%
class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    
    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, N, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, N, **kwargs)        
        # Add an output layer
        self.fc = nn.Linear(N, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output

#%%
path = 'src/loader/'
file_path = path + 'chaotic_rnn_200x5_tau25_dt10_g_3_btw_sparsity_01.pth'


def chaotic_rnn_loader(n_trial, n_list, T = 500, file_path = file_path):

    n_area = len(n_list)
    N = 200*n_area
    input_size = 100
    net = RNNNet(input_size, N, 2, tau=25, dt = 10)
    net.load_state_dict(torch.load(file_path, weights_only=True))
    net.eval()

    x = torch.zeros(T, n_trial, input_size)
    
    out, activity = net(x)


    factors = 1+activity.permute(1,0,2).detach().numpy() #(n_trial, T, n_factors)

    n = np.sum(n_list)
    
    area_ind_list = np.concatenate([np.ones(n_list[i])*i for i in range(n_area)]).astype(int)
    spike_data = np.zeros((n_trial, T, n))
    fr = np.zeros((n_trial, T, n))

    n_factors = 200

    for i in range(n_area):
        w = random(n_factors, n_list[i], density=0.02).toarray()
        fr[:,:,np.sum(n_list[:i]):np.sum(n_list[:i+1])] = factors[:,:,n_factors*i: n_factors*(i+1)] @ w

    #normalized fr
    epsilon = 1e-10
    min_fr = np.min(fr, axis=(0,1))[None,None,:]
    max_fr = np.max(fr, axis=(0,1))[None,None,:]
    #fr_norm = (fr-min_fr)/(max_fr-min_fr + epsilon)*2
    #fr_norm[np.isnan(fr_norm)] = 0
    
    fr_norm = (fr-min_fr)/(max_fr-min_fr + epsilon)*6-3 #from -3 to 3
    fr_norm[np.isnan(fr_norm)] = -3

    spike_data = np.random.poisson(np.exp(fr_norm))

    return spike_data[:,100:,:], area_ind_list, fr_norm[:,100:,:], factors[:,100:,:]

#%%    
class BaseDatasetRNN(torch.utils.data.Dataset):
    def __init__(self, spike_data, choice,
                 area_ind_list, session_ind, fr, factors, spike_data_full, area_ind_list_full):
              
        self.spike_data = spike_data # K x T x N
        self.area_ind_list = area_ind_list  # list of area_ind for each neuron
        self.session_ind = session_ind

        self.behavior_data = choice[:, None, None]*np.ones((1, spike_data.shape[1], 1)) # K x T x 1
        self.fr = np.exp(fr)
        self.factors = factors
        self.spike_data_full = spike_data_full
        self.area_ind_list_full = area_ind_list_full

        K, T, N = self.spike_data.shape
        self.N = N
        self.T = T
        self.K = K

    def _preprocess_rnn_data(self, idx):
        # idx is the trial index
        binned_spikes_data = self.spike_data[idx].astype(np.float32) # T x N
        fr_data = self.fr[idx].astype(np.float32) # T x N
        factors_data = self.factors[idx].astype(np.float32) # T x n_factors

        spikes_data_full = self.spike_data_full[idx].astype(np.float32) # T x N_all

        spikes_timestamps = np.arange(self.T)  

        target_behavior = self.behavior_data[idx] # T x D

        trial_type = 0

        return {
                    "spikes_data": binned_spikes_data,
                    "spikes_timestamps": spikes_timestamps,
                    "target": target_behavior,
                    "neuron_regions": self.area_ind_list,
                    "eid": self.session_ind,
                    "fr": fr_data, #full fr
                    "factors": factors_data, #full factors
                    "spikes_data_full": spikes_data_full, 
                    "neuron_regions_full": self.area_ind_list_full,
                    "trial_type": trial_type
                }


    def __len__(self):
        return self.K
    
    def __getitem__(self, idx):
        return self._preprocess_rnn_data(idx)

#%%
def make_chaotic_rnn_loader(session_ind_list, batch_size, seed=42, distributed=False, rank=0, world_size=1):
    datasets = {'train': [], 'val': [], 'test': []}
    data_loader = {}
    num_neurons = []
    num_trials = {'train': [], 'val': [], 'test': []}
    area_ind_list_list = []
    record_info_list = []
    
    #path = '/work/hdd/bdye/jxia4/data/generated_before_chaotic_rnn/5_area_chaotic_rnn_g3_factor_200/'
    path = 'data/generated_before_chaotic_rnn/5_area_chaotic_rnn_g3_factor_200_log_fr_neg3_to_3/'

    n_area = 5
    os.makedirs(path, exist_ok=True)

    generated_session_ind_list = [int(file.split('_')[-1].split('.')[0]) for file in os.listdir(path)]  

    for session_ind in session_ind_list:
        session_path = path + f'chaotic_rnn_{session_ind}.pkl'
        np.random.seed(session_ind+100)
        if session_ind in generated_session_ind_list:
            spike_data, area_ind_list, fr, factors, n_list, K, omit_region, train_ind, val_ind, test_ind = pickle.load(open(session_path, 'rb'))
            print('Loading existing data session ', session_ind)
        else:
            n_list = np.random.randint(low=20, high=60, size=n_area)
            K = np.random.randint(200, 300)
            #generate spike data from all areas
            spike_data, area_ind_list, fr, factors = chaotic_rnn_loader(K, n_list, T = 500)
            #randomly omit one or two region
            if (np.random.rand() < 0.5) and (n_area > 3):
                omit_region = np.random.choice(n_area, 2, replace=False)
            else:
                omit_region = np.random.choice(n_area, 1)

            #np.random.seed(seed)
            #get train/valid/test trial_ind
            indices = np.arange(K)
            np.random.shuffle(indices)
            train_ind, val_ind, test_ind = np.split(indices, [int(.6*K), int(.8*K)])
            
            pickle.dump((spike_data, area_ind_list, fr, factors, n_list, K, omit_region, train_ind, val_ind, test_ind), open(session_path, 'wb'))

        choice = np.zeros((K,))
        #omit data from one or two region
        record_neuron_flag = np.ones((np.sum(n_list),), dtype=bool)
        for region in omit_region:
            record_neuron_flag &= (area_ind_list != region) 

        record_info = {'gt_n': n_list, 'omit_region': omit_region}
        record_info_list.append(record_info)

        spike_data_full = spike_data.copy()
        area_ind_list_full = area_ind_list.copy()

        spike_data = spike_data[:,:,record_neuron_flag==1]
        area_ind_list = area_ind_list[record_neuron_flag==1]
        
        for ind, name in zip([train_ind, val_ind, test_ind], ['train', 'val', 'test']):
            dataset = BaseDatasetRNN(spike_data[ind], choice[ind], area_ind_list, session_ind, fr[ind], factors[ind], spike_data_full[ind], area_ind_list_full)
            num_trials[name].append(len(ind))
            datasets[name].append(dataset)

        num_neurons.append(dataset.N) # train, val, test should have the same number of neurons; so it's fine to just append the last one
        area_ind_list_list.append(area_ind_list)

    print('num_neurons: ', num_neurons)
    print('num_trials: ', num_trials)
    
    for name in ['train', 'val', 'test']:
        if batch_size < sum(num_trials[name]):
            dataset_list = datasets[name]
            if name != 'test':
                data_loader[name] =  DatasetDataLoader(dataset_list, batch_size, seed=seed, distributed=distributed, rank=rank, world_size=world_size, drop_last=True)
            else:
                data_loader[name] =  DatasetDataLoader(dataset_list, batch_size= sum(num_trials[name]), seed=seed, distributed=distributed, rank=rank, world_size=world_size, drop_last=False) #for test, the batch size is the total number of trials
            print('Succesfully constructing the dataloader for ', name)

    return data_loader, num_neurons, datasets, area_ind_list_list, record_info_list

def main():
    data_loader, num_neurons, _, area_ind_list_list, record_info_list = make_chaotic_rnn_loader(np.arange(10), 12)

    print('num_neurons: ', num_neurons)
    print('area_ind_list_list: ', area_ind_list_list)
    print('record_info_list: ', record_info_list)

    for batch in data_loader['test']:
        print(batch['spikes_data'].shape)
        print(batch['target'].shape)
        
    breakpoint()

if __name__ == "__main__":
    main()

# %%
