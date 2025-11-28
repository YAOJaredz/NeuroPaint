import os, sys
sys.path.append('src')
import pandas as pd
from pynwb import NWBHDF5IO
import pickle, json
import torch
import numpy as np
import glob
from one.api import ONE
from utils.ibl_data_utils import *
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from loader.loader_utils import DatasetDataLoader
from constants import DATA_INFO_PATH, IBL_AREAOI, IBL_DATA_PATH, DATA_PATH, DATASET_HELDOUT_INFO_PATH

#%%
# load session_info.csv as pandas dataframe
def load_session_data(session_ind, session_idx):
    '''
    now we load all trials
    Note that the loaded/save data contain neurons from all recorded areas, including those not in the area of interest

    return:

    shape of returned data:
    spike_data: N x T x K
    behavior_data: K x T x D
    area_ind_list: N 
    trial_type: K
    '''

    with open(DATA_INFO_PATH / "ibl_eids.txt") as file:
        eids = [line.rstrip() for line in file]

    print(f"EID {session_ind}")

    files = sorted(glob.glob(str(IBL_DATA_PATH / "session_ind_*.pickle")))
    session_ind_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
    if session_idx not in session_ind_list:
        params = {
            "interval_len": 2, "binsize": 0.01, "single_region": False, 
            "align_time": 'stimOn_times', "time_window": (-.5, 1.5), "fr_thresh": 0.5
        }
        beh_names = ["choice", "reward", "block", "wheel-speed", "whisker-motion-energy"]
        DYNAMIC_VARS = list(filter(lambda x: x not in ["choice", "reward", "block"], beh_names))

        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international", 
            silent=True,
            cache_dir=DATA_PATH / "ibl_raw/"
        )
        neural_dict, behave_dict, meta_dict, trials_dict, _ = prepare_data(
            one, session_ind, params, n_workers=1
        )
        regions, beryl_reg = list_brain_regions(neural_dict, **params)
        region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

        bin_spikes, clusters_used_in_bins = bin_spiking_data(
            region_cluster_ids, 
            neural_dict, 
            trials_df=trials_dict["trials_df"], 
            n_workers=1, 
            **params
        )
        print(f"Binned Spike Data: {bin_spikes.shape}")

        bin_beh, beh_mask = bin_behaviors(
            one, 
            session_ind, 
            DYNAMIC_VARS, 
            trials_df=trials_dict["trials_df"], 
            allow_nans=True, 
            n_workers=1, 
            **params,
        )
        bin_lfp = None
        try:
            align_bin_spikes, align_bin_beh, align_bin_lfp, target_mask, bad_trial_idxs = align_data(
                bin_spikes, 
                bin_beh, 
                bin_lfp, 
                list(bin_beh.keys()), 
                trials_dict["trials_mask"], 
            )
        except ValueError as e:
            print(f"Skip EID {session_ind} due to error: {e}")
            return None, None, None, None, None, None

        spike_data = align_bin_spikes.swapaxes(0, 2)
        area_ind_list = meta_dict["cluster_regions"]
        behavior = None
        is_left = None
        is_ALM = None
        trial_type = align_bin_beh["choice"]

        # Save data
        with open(IBL_DATA_PATH / f"session_ind_{session_idx}.pickle", 'wb') as f:
            pickle.dump([spike_data, behavior, area_ind_list, is_left, is_ALM, trial_type], f)
    else:
        print(f"Loading EID {session_ind} from cached data")
        with open(files[session_ind_list.index(session_idx)], 'rb') as f:
            spike_data, behavior, area_ind_list, is_left, is_ALM, trial_type = pickle.load(f)

    area_ind_list = list(area_ind_list)

    return spike_data, behavior, area_ind_list, is_left, is_ALM, trial_type


class BaseDataset(torch.utils.data.Dataset):
    '''
    all the loaded neurons in the dataset belongs to areas in the areaoi_ind
    the spike_data in the input contains neurons from all of the areas
    but the self.spike_data only contain neurons in the areas of interest
    '''
    def __init__(
        self, 
        spike_data, 
        behavior_data, 
        trial_type_data,
        area_ind_list, 
        is_left, 
        areaoi_ind, 
        session_ind, 
        spike_data_full, 
        area_ind_list_full, 
        is_left_full,
        ):
        
        neuronoi_ind = np.array([], dtype=int)
        for area_ind in areaoi_ind:
            neuronoi_ind_tmp = np.where(np.array(area_ind_list) == area_ind)[0]
            if len(neuronoi_ind_tmp) >= 5: # only include areas with at least 5 neurons
                neuronoi_ind = np.concatenate((neuronoi_ind, neuronoi_ind_tmp))

        self.spike_data = np.swapaxes(spike_data[neuronoi_ind], 0, 2) # K x T x N
        self.area_ind_list = np.array(area_ind_list)[neuronoi_ind].tolist()  # list of area_ind for each neuron
        self.areaoi_ind = areaoi_ind # area_ind of the area of interest
        self.session_ind = session_ind
        self.trial_type_data = trial_type_data.flatten().tolist() # K
        self.behavior_data = behavior_data # K x T x D
        self.is_left = is_left
        if self.is_left is not None:
            self.is_left = self.is_left[neuronoi_ind] # list of is_left for each neuron, flag for whether the neuron is from left or right hemisphere
        self.spike_data_full = np.swapaxes(spike_data_full, 0, 2) # K x T x N_full #contains all neurons, including the heldout ones and the ones not in the area of interest
        self.area_ind_list_full = area_ind_list_full
        self.is_left_full = is_left_full

        K, T, N = self.spike_data.shape
        self.N = N
        self.T = T
        self.K = K

    def _preprocess_ibl_data(self, idx):
        
        binned_spikes_data = self.spike_data[idx].astype(np.float32) # T x N

        spikes_timestamps = np.arange(self.T)  # maybe can be changed to behavior_dict[type]['ts'] later

        if self.behavior_data is not None:
            target_behavior = self.behavior_data[idx] # T x D
        else:
            target_behavior = np.array([0] * self.T)

        neuron_regions = np.array(self.area_ind_list).T

        neuron_regions_full = np.array(self.area_ind_list_full).T

        spike_data_full = self.spike_data_full[idx].astype(np.float32)
        
        return {
            "spikes_data": binned_spikes_data,
            "spikes_timestamps": spikes_timestamps,
            "target": target_behavior,
            "neuron_regions": neuron_regions,
            "spikes_data_full": spike_data_full,
            "neuron_regions_full": neuron_regions_full,
            "is_left": self.is_left if self.is_left is not None else np.array([0] * self.N),
            "is_left_full": self.is_left_full if self.is_left_full is not None else np.array([0] * self.N),
            "eid": self.session_ind,
            "trial_type": self.trial_type_data[idx]                   
        }

    def __len__(self):
        return self.K
    
    def __getitem__(self, idx):
        return self._preprocess_ibl_data(idx)



#%%
def update_area_ind_list_and_areaoi_ind(area_ind_list, areaoi):
    """
    areaoi: indices of the area of interest (B,)
    area_ind_list: indices of the area of interest (N,) 
    """

    for re_idx, re_name in enumerate(area_ind_list):
        if "DG" in re_name:
            area_ind_list[re_idx] = "DG"
        elif ("VISa" in re_name) or ("VISam" in re_name):
            area_ind_list[re_idx] = "VISa"
        elif "LGd" in re_name:
            area_ind_list[re_idx] = "LGd"
        elif "VPM" in re_name:
            area_ind_list[re_idx] = "VPM"

    region_neuron_count = {region: 0 for region in areaoi}
    for region in area_ind_list:
        if region in areaoi:
            region_neuron_count[region] += 1

    print('Brain region summary: ')
    print(region_neuron_count)

    return area_ind_list, region_neuron_count
    
#%%
def make_loader(
    session_ind_list, 
    batch_size, 
    seed=42, 
    distributed=False, 
    rank=0, 
    world_size=1
):
    path = DATA_INFO_PATH

    areaoi = IBL_AREAOI
    region_to_ind = {region: i for i, region in enumerate(areaoi)}
    
    datasets = {'train': [], 'val': [], 'test': []}
    num_trials = {'train': [], 'val': [], 'test': []}
    data_loader = {}
    num_neurons = []
    area_ind_list_list = []
    heldout_info_list = []
    trial_type_dict = {'left': -1, 'right':1} #right and left might be swapped; need to verify later 

    np.random.seed(seed)

    region_neuron_count_dict = {}
    for session_idx, session_ind in tqdm(enumerate(session_ind_list), total=len(session_ind_list)):

        print(session_ind)
        
        spike_data, behavior_data, area_ind_list, is_left, is_ALM, trial_type = load_session_data(session_ind, session_idx)
                
        area_ind_list, region_neuron_count = update_area_ind_list_and_areaoi_ind(area_ind_list, areaoi)
        region_neuron_count_dict[session_idx] = region_neuron_count

        # randomly hold out data from 1 area of interest for some of the sessions
        record_neuron_flag = np.ones(len(area_ind_list), dtype=bool)
        
        # choose only the neurons in the area of interest (grouped)
        area_ind_unique = np.unique(area_ind_list)
        areaoi_ind_exist = np.intersect1d(area_ind_unique, areaoi)

        np.random.seed(session_idx+1000)
        heldout_region = np.random.choice(areaoi_ind_exist, 1)[0]
        record_neuron_flag &= (np.array(area_ind_list) != heldout_region)
        
        heldout_info = {'session_ind': session_ind, 'heldout_region_name': heldout_region, 'heldout_region_ind': region_to_ind[heldout_region]}
        heldout_info_list.append(heldout_info)

        # add new regions to region_to_ind
        for region in np.unique(area_ind_list):
            if region not in region_to_ind:
                region_to_ind[region] = len(region_to_ind)
        print("Updated region index dict: ", region_to_ind)

        spike_data_full = spike_data.copy()
        area_ind_list_full = area_ind_list.copy()

        print("Spike data shape: ", spike_data_full.shape)
        
        spike_data = spike_data[record_neuron_flag==1]
        area_ind_list = np.array(area_ind_list)[record_neuron_flag==1].tolist()

        print("Spike data shape after removing unrecorded areas: ", spike_data.shape)

        np.random.seed(seed)
        K = len(trial_type)
        indices = np.arange(K)
        np.random.shuffle(indices)
        train_ind, val_ind, test_ind = np.split(indices, [int(.6*K), int(.8*K)])

        area_ind_list = [region_to_ind[region] for region in area_ind_list]
        area_ind_list_full = [region_to_ind[region] for region in area_ind_list_full]
        areaoi_ind = [region_to_ind[region] for region in areaoi]
        
        for ind, name in zip([train_ind, val_ind, test_ind], ['train', 'val', 'test']):

            is_left_full = None

            dataset = BaseDataset(
                spike_data[...,ind], 
                behavior_data[ind] if behavior_data is not None else None, 
                trial_type[ind], 
                area_ind_list, 
                is_left, 
                areaoi_ind, 
                session_idx, # session_ind 
                spike_data_full[...,ind], 
                area_ind_list_full, 
                is_left_full,
            )
            num_trials[name].append(len(ind))
            datasets[name].append(dataset)

        num_neurons.append(dataset.N) # train, val, test should have the same number of neurons; so it's fine to just append the last one
        area_ind_list_list.append(dataset.area_ind_list) # make sure the area_ind_list only contain neurons in the area of interest 
        
    print('num_neurons: ', num_neurons)
    print('num_trials: ', num_trials)
    
    for name in ['train', 'val', 'test']:
        if batch_size < sum(num_trials[name]):
            dataset_list = datasets[name]
            if name != 'test':
                data_loader[name] =  DatasetDataLoader(dataset_list, batch_size, seed=seed, distributed=distributed, rank=rank, world_size=world_size)
            else:
                data_loader[name] =  DatasetDataLoader(dataset_list, batch_size= sum(num_trials[name]), seed=seed, distributed=distributed, rank=rank, world_size=world_size) #for test, the batch size is the total number of trials
            print('Succesfully constructing the dataloader for ', name)

    with open(DATASET_HELDOUT_INFO_PATH / f'ibl_heldout_info_{len(session_ind_list)}.json', 'w') as f:
        json.dump(heldout_info_list, f, indent=4)
    
    # np.save("/work/hdd/bdye/jxia4/data/tables_and_infos/ibl_region_to_ind.npy", region_to_ind)
    # np.save("/work/hdd/bdye/jxia4/data/tables_and_infos/ibl_region_neuron_count.npy", region_neuron_count_dict)
    
    return data_loader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict


#%%
def main():

    with open(f"data/tables_and_infos/ibl_eids.txt") as file:
        eids = [line.rstrip() for line in file]

    data_loader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict = make_loader(eids, 12)

    print(heldout_info_list)

    for batch in data_loader['train']:
        # print(batch.keys())
        print(batch["spikes_data"].shape)
        print(batch["spikes_data_full"].shape)
        print(len(batch['neuron_regions']))
        print(len(batch['neuron_regions_full']))
        print(np.unique(batch['neuron_regions']))
        print(np.unique(batch['neuron_regions_full']))
        print(batch['is_left'][0])
        break

    breakpoint()

if __name__ == "__main__":
    main()

# %%




