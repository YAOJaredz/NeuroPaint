# %%
import sys
sys.path.append('src')
from utils.svoboda_data_utils import *
import pandas as pd
from pynwb import NWBHDF5IO
import pickle
import torch
import numpy as np
import glob
from loader.loader_utils import DatasetDataLoader
#%%
# load session_info.csv as pandas dataframe
def load_session_data(session_ind, trial_type_dict):
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
    files = sorted(glob.glob('/work/hdd/bdye/jyao7/data/loaded_before_unbalanced_is_ALM_is_left/session_ind_*.pickle'))
    session_ind_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
    if session_ind in session_ind_list:
        with open(files[session_ind_list.index(session_ind)], 'rb') as f:
            return pickle.load(f)

    path = '/work/hdd/bdye/jyao7/data/tables_and_infos/'

    session_info = pd.read_csv(path + 'session_info.csv')
    file_name = session_info['session_name'][session_ind]
    print(session_info.iloc[session_ind])

    if session_info.iloc[session_ind]['opto_onset']>0.5:
        opto_timing = 'late'
    else:
        opto_timing = 'early'


    io = NWBHDF5IO(path + file_name, mode="r")
    nwbfile = io.read()

    # get the spike train
    spike_data_combine_all, n_trial_all, licks_dict, behavior_dict, units, ids_sort_by_area, opto_type_sorted_dict, sorted_area, area_value, area_value_dict, is_left, is_ALM = main_get_spike_trains(nwbfile)
    
    spike_data_unbalanced = []
    trial_type_unbalanced = []

    for type in ['hit', 'miss', 'ignore']:

        if (n_trial_all[type]['left'] == 0) and (n_trial_all[type]['right'] == 0):
            continue

        spike_data = spike_data_combine_all[type]
        trial_type_ind = [trial_type_dict[type + '_left']] * n_trial_all[type]['left'] + [trial_type_dict[type + '_right']] * n_trial_all[type]['right']
        spike_data_unbalanced.append(spike_data)
        trial_type_unbalanced.append(trial_type_ind)


    for type in ['opto_hit', 'opto_miss', 'opto_ignore']:
        
        if (n_trial_all[type]['left'] == 0) and (n_trial_all[type]['right'] == 0):
            continue
        
        spike_data = spike_data_combine_all[type]
        spike_data_unbalanced.append(spike_data)

        trial_outcome = type.split('_')[1] # type is hit, miss, or ignore
        opto_trial_type = opto_type_sorted_dict[trial_outcome] # a list of np array with 1 element

        for opto_trial in opto_trial_type[:n_trial_all[type]['left']]:
            trial_type_name = 'opto_' + opto_timing + '_' + str(opto_trial[0]) + '_' + trial_outcome + '_left'
            trial_type_unbalanced.append([trial_type_dict[trial_type_name]])

        for opto_trial in opto_trial_type[n_trial_all[type]['left']:]:
            trial_type_name = 'opto_' + opto_timing + '_' + str(opto_trial[0]) + '_' + trial_outcome + '_right'
            trial_type_unbalanced.append([trial_type_dict[trial_type_name]])
            
    spike_data_unbalanced = np.concatenate(spike_data_unbalanced, axis=2)
    trial_type_unbalanced = np.concatenate(trial_type_unbalanced)
    
    #load region_info_summary.pkl
    with open(path + 'region_info_summary.pkl', 'rb') as f:
        [brain_region_list, session_by_region, session_by_region_n, junk] = pickle.load(f)

    # turn the area_value (area ind created by sorting within a session) into a list of area index (consistent across sessions)
    area_value = np.array(area_value)
    area_ind_list = np.zeros_like(area_value)
    for area, value in area_value_dict.items():
        area_ind = np.where(brain_region_list == area)[0][0]
        area_ind_list[area_value==value] = area_ind

    # get behavior data as K x T X D
    behavior_data_concat_dict = {'jaw': [], 'nose': [], 'tongue': []}

    for name in ['jaw', 'nose', 'tongue']:
        for trial_outcome in ['hit', 'miss', 'ignore', 'opto_hit', 'opto_miss', 'opto_ignore']:
            for side in ['left', 'right']:
                if n_trial_all[trial_outcome][side] == 0:
                    continue
                behavior_data_concat_dict[name].append(behavior_dict[trial_outcome][side][name])
        
        behavior_data_concat_dict[name] = np.concatenate(behavior_data_concat_dict[name], axis=0)

    behavior_unbalanced = np.concatenate((behavior_data_concat_dict['jaw'], behavior_data_concat_dict['nose'], behavior_data_concat_dict['tongue']), axis=2)

    with open('/work/hdd/bdye/jyao7/data/loaded_before_unbalanced_is_ALM_is_left/session_ind_{}.pickle'.format(session_ind), 'wb') as f:
        pickle.dump([spike_data_unbalanced, behavior_unbalanced, area_ind_list, is_left, is_ALM, trial_type_unbalanced], f)

    print('shape of spike_data_unbalanced: ')
    print(spike_data_unbalanced.shape)
    
    print('shape of behavior_unbalanced: ')
    print(behavior_unbalanced.shape)

    print('shape of trial_type_unbalanced: ')
    print(trial_type_unbalanced.shape)

    return spike_data_unbalanced, behavior_unbalanced, area_ind_list, is_left, is_ALM, trial_type_unbalanced

#%%
class BaseDataset(torch.utils.data.Dataset):
    '''
    all the loaded neurons in the dataset belongs to areas in the areaoi_ind
    the spike_data in the input contains neurons from all of the areas
    but the self.spike_data only contain neurons in the areas of interest
    '''

    def __init__(self, spike_data, behavior_data, trial_type_data,
                 area_ind_list, is_left, areaoi_ind, 
                 session_ind, spike_data_full, area_ind_list_full, is_left_full):
        
        neuronoi_ind = np.array([], dtype=int)
        for area_ind in areaoi_ind:
            neuronoi_ind_tmp = np.where(area_ind_list == area_ind)[0]
            if len(neuronoi_ind_tmp)>=5:                                    # only include areas with at least 5 neurons
                neuronoi_ind = np.concatenate((neuronoi_ind, neuronoi_ind_tmp))

        self.spike_data = np.swapaxes(spike_data[neuronoi_ind], 0, 2) # K x T x N
        self.area_ind_list = area_ind_list[neuronoi_ind]  # list of area_ind for each neuron
        self.is_left = is_left[neuronoi_ind] # list of is_left for each neuron, flag for whether the neuron is from left or right hemisphere
        self.areaoi_ind = areaoi_ind # area_ind of the area of interest
        self.session_ind = session_ind
        self.behavior_data = behavior_data # K x T x D
        self.trial_type_data = trial_type_data # K

        self.spike_data_full = np.swapaxes(spike_data_full, 0, 2) # K x T x N_full #contains all neurons, including the heldout ones and the ones not in the area of interest
        self.area_ind_list_full = area_ind_list_full
        self.is_left_full = is_left_full

        K, T, N = self.spike_data.shape
        self.N = N
        self.T = T
        self.K = K

    def _preprocess_svoboda_data(self, idx):
        
        binned_spikes_data = self.spike_data[idx].astype(np.float32) # T x N

        spikes_timestamps = np.arange(self.T)  # maybe can be changed to behavior_dict[type]['ts'] later

        target_behavior = self.behavior_data[idx] # T x D
        neuron_regions = self.area_ind_list

        spike_data_full = self.spike_data_full[idx].astype(np.float32)
        
        return {
                    "spikes_data": binned_spikes_data,
                    "spikes_timestamps": spikes_timestamps,
                    "target": target_behavior,
                    "neuron_regions": neuron_regions,
                    "is_left": self.is_left,
                    "eid": self.session_ind,
                    "spikes_data_full": spike_data_full,
                    "neuron_regions_full": self.area_ind_list_full,
                    "is_left_full": self.is_left_full,
                    "trial_type": self.trial_type_data[idx]                    
                }


    def __len__(self):
        return self.K
    
    def __getitem__(self, idx):
        return self._preprocess_svoboda_data(idx)

#%%
def update_area_ind_list_and_areaoi_ind(area_ind_list, areaoi, brain_region_list, is_ALM):
    areaoi_ind = np.zeros((len(areaoi),), dtype=int)
    #ALM
    area_ind = len(brain_region_list)-1
    area_ind_list[is_ALM==1] = area_ind
    areaoi_ind[0] = area_ind

    #lump layers
    ind_lump_layers = [1,2,-1]
    for ind in ind_lump_layers:
        nameoi = areaoi[ind]
        #print(nameoi)
        area_ind_lump_list = []
        for area_ind, area_name in enumerate(brain_region_list):
            if nameoi in area_name:
                area_ind_lump_list.append(area_ind)
                #print(area_name)
        
                area_ind_list[area_ind_list==area_ind] = area_ind_lump_list[0]
        
        areaoi_ind[ind] = area_ind_lump_list[0]

    #lump areas
    #include Pallidum, Globus pallidus, external segment

    ind_lump_areas = [3, 4, 5]

    areas_list = [['Pallidum', 'Globus pallidus, external segment'],
                ['Striatum', 'Caudoputamen'],
                ['Ventral anterior-lateral complex of the thalamus', 'Ventral medial nucleus of the thalamus']]

    for ind, areas in zip(ind_lump_areas, areas_list):
        area_ind_lump_list = []
        #print(areaoi[ind])

        for area in areas:
            area_ind = np.where(brain_region_list == area)[0][0]
            area_ind_lump_list.append(area_ind)

            area_ind_list[area_ind_list==area_ind] = area_ind_lump_list[0]
            #print(brain_region_list[area_ind])
            
        areaoi_ind[ind] = area_ind_lump_list[0]

    #the rest
    ind_normal = [-2]
    for ind in ind_normal:
        area_ind = np.where(brain_region_list == areaoi[ind])[0][0]
        areaoi_ind[ind] = area_ind

    return areaoi_ind, area_ind_list
    
#%%
def make_loader(session_ind_list, batch_size, include_opto = False, seed=42, distributed=False, rank=0, world_size=1):

    path = '/work/hdd/bdye/jyao7/data/tables_and_infos/'

    #load region_info_summary.pkl
    with open(path + 'region_info_summary.pkl', 'rb') as f:
        [brain_region_list, session_by_region, session_by_region_n, junk] = pickle.load(f)

    brain_region_list = np.append(brain_region_list, 'ALM') #add ALM to the list

    areaoi = ['ALM', #according to is_ALM
            'Orbital area, lateral part', #lump layers
            'Orbital area, ventrolateral part', #lump layers
            'Pallidum', #include Pallidum, Globus pallidus, external segment
            'Striatum', #include Striatum, Caudoputamen
            'VAL-VM', #include Ventral anterior-lateral complex of the thalamus, Ventral medial nucleus of the thalamus
            'Midbrain reticular nucleus',
            'Superior colliculus, motor related'#lump layers
            ]

    #areaoi_ind = [np.where(brain_region_list == area)[0][0] for area in areaoi]
    
    datasets = {'train': [], 'val': [], 'test': []}
    data_loader = {}
    num_neurons = []
    num_trials = {'train': [], 'val': [], 'test': []}
    area_ind_list_list = []
    heldout_info_list = []

    trial_type_dict = {}
    opto_type_ind_list = []

    i = 0
    for name1 in ['left', 'right']:
        for name2 in ['hit', 'miss', 'ignore']:
            for name3 in ['', 'opto_early_4_', 'opto_late_4_', 'opto_early_5_', 'opto_late_5_', 'opto_early_6_', 'opto_late_6_']:
                trial_type_dict[name3 + name2 + '_' + name1] = i
                
                if 'opto' in name3:
                    opto_type_ind_list.append(i)

                i += 1

    np.random.seed(seed)
    for session_ind in session_ind_list:
        print(session_ind)
        spike_data_unbalanced, behavior_unbalanced, area_ind_list, is_left, is_ALM, trial_type_unbalanced = load_session_data(session_ind, trial_type_dict)
        is_left = is_left.astype(np.int64)
        
        areaoi_ind, area_ind_list = update_area_ind_list_and_areaoi_ind(area_ind_list, areaoi, brain_region_list, is_ALM)
        #breakpoint()
        #exclude all the opto trials
        if not include_opto:
            flag_non_opto = np.ones(len(trial_type_unbalanced), dtype=bool)
            for ind in opto_type_ind_list:
                flag_non_opto &= (trial_type_unbalanced != ind)
            
            spike_data_unbalanced = spike_data_unbalanced[:,:,flag_non_opto]
            behavior_unbalanced = behavior_unbalanced[flag_non_opto]
            trial_type_unbalanced = trial_type_unbalanced[flag_non_opto]

        #randomly hold out data from 1 area of interest for each session
        record_neuron_flag = np.ones(len(area_ind_list), dtype=bool)
        
        area_ind_unique = np.unique(area_ind_list)
        areaoi_ind_exist = np.intersect1d(area_ind_unique, areaoi_ind)

        if len(areaoi_ind_exist) <= 1:
            print(f'less or equal to 1 area of interest in session {session_ind}, skip this session')
            continue


        np.random.seed(session_ind+1000)
        heldout_region = np.random.choice(areaoi_ind_exist,1)[0]
        record_neuron_flag &= (area_ind_list != heldout_region)
        
        heldout_info = {'session_ind': session_ind, 'heldout_region_ind': heldout_region, 'heldout_region_name': brain_region_list[heldout_region]}
        heldout_info_list.append(heldout_info)

        spike_data_full = spike_data_unbalanced.copy()
        area_ind_list_full = area_ind_list.copy()
        is_left_full = is_left.copy()
        
        spike_data_unbalanced = spike_data_unbalanced[record_neuron_flag==1]
        area_ind_list = area_ind_list[record_neuron_flag==1]
        is_left = is_left[record_neuron_flag==1]

        np.random.seed(seed)
        #get train/valid/test trial_ind #here I only load hit left trials. Later we can add more types of trials
        K = len(trial_type_unbalanced)
        indices = np.arange(K)
        np.random.shuffle(indices)
        train_ind, val_ind, test_ind = np.split(indices, [int(.6*K), int(.8*K)])
        
        for ind, name in zip([train_ind, val_ind, test_ind], ['train', 'val', 'test']):

            dataset = BaseDataset(spike_data_unbalanced[:,:,ind], behavior_unbalanced[ind], trial_type_unbalanced[ind], area_ind_list, is_left, areaoi_ind, session_ind, spike_data_full[:,:,ind], area_ind_list_full, is_left_full)
            num_trials[name].append(len(ind))
            datasets[name].append(dataset)

        num_neurons.append(dataset.N) # train, val, test should have the same number of neurons; so it's fine to just append the last one
        area_ind_list_list.append(dataset.area_ind_list) #make sure the area_ind_list only contain neurons in the area of interest 
        
        #print('session_ind: ', session_ind)
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
    
    #breakpoint()
    return data_loader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict


#%%
def main():
    session_order = pickle.load(open('/work/hdd/bdye/jyao7/data/tables_and_infos/session_order.pkl', 'rb'))

    eids = np.sort(session_order[:40])

    print(eids)

    #exclude the following sessions [ 13,  63, 136, 153, 160] at indices [12, 58,123,139,146], these are 5 sessions with 0 areas of interest after I held-out 1 area.
    #eids = np.delete(eids, [ 12,  58, 123, 139, 146]) #for 160 sessions

    #breakpoint()

    data_loader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict = make_loader(eids, 12, include_opto=False)

    print(heldout_info_list)

    for batch in data_loader['train']:
        print(batch['spikes_data'].shape)
        print(batch['target'].shape)
        print(batch['trial_type'].shape)
        breakpoint()
        
        break

    print(heldout_info_list)

    breakpoint()

if __name__ == "__main__":
    main()

# %%
