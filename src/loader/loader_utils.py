from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch
import numpy as np


class DatasetDataLoader:
    def __init__(
        self, 
        datasets_list, 
        batch_size, 
        seed=42, 
        distributed=False, 
        rank=0, 
        world_size=1,
        drop_last=False
    ):
        self.seed = seed
        self.distributed = distributed
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        np.random.seed(seed)
        
        self.loaders = []
        for dataset in datasets_list:
            if self.distributed:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=drop_last)
            else:
                sampler = RandomSampler(dataset)

            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, generator=generator)
            self.loaders.append(loader)
    
    def set_epoch(self, epoch):
        if self.distributed:
            for loader in self.loaders:
                if hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)

    def __iter__(self):
        self.iter_loaders = [iter(loader) for loader in self.loaders]
        self.loader_order = np.random.choice(len(self.loaders), size=len(self.loaders), replace=False)
        self.ind = 0
        self.loader_exhausted = [False] * len(self.loaders)
        return self

    def __next__(self):
        if all(self.loader_exhausted):
            raise StopIteration

        loader_ind = self.loader_order[self.ind]
        self.ind = (self.ind + 1) % len(self.loader_order)  # Wrap around to the start
        try:
            return next(self.iter_loaders[loader_ind])
        except StopIteration:
            # If the DataLoader for this dataset is exhausted, mark it as exhausted
            self.loader_exhausted[loader_ind] = True
            # Try to get the next batch from the next DataLoader
            return self.__next__()

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)
    