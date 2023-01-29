import json, yaml
import os
import h5py as h5
import numpy as np
import torch
from torch.utils import data


def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    return train_data,test_data

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def ReverseNormChallenge(e,voxels,layer_es,emax=1000,emin=1,max_deposit=2.3,logE=False):
    '''Revert the transformations applied to the training set'''
    #shape=voxels.shape
    alpha = 1e-6
    if logE:
        energy = emin*(emax/emin)**e
    else:
        energy = e
    exp = np.exp(voxels)    
    x = exp/(1+exp)
    data = (x-alpha)/(1 - 2*alpha)
    data = data*(np.expand_dims(layer_es,(2,3))) #+ np.full_like(data, 1e-16))
    data = data.reshape(voxels.shape[0],-1)
    return energy,data

def DataLoaderDataset2(file_name,nevts,emax=1000,emin=1,max_deposit=2.3,logE=False,inc_noise=True):
    with h5.File(file_name,"r") as h5f:
        e = h5f['incident_energies'][0:int(nevts)].astype(np.float32)/1000.0
        shower = h5f['showers'][0:int(nevts)].astype(np.float32)/1000.0 #probably NOT in MeV
    
    shower = shower.reshape(-1,9,8,8)
    energy = np.sum(shower, (2,3))
    if inc_noise: energy += np.random.uniform(0,1e-10,size=energy.shape)
    shower = np.ma.divide(shower, np.expand_dims(energy,(2,3))).filled(0)
    if inc_noise: shower += np.random.uniform(0,1e-9,size=shower.shape) #less noise? (trial 1)
    #print(shower[0])
    
    print(np.sum(energy[0]))
    print(e[0])
    
    #print(shower[0])
    shower = shower.reshape(shower.shape[0],-1)

    alpha = 1e-6
    x = alpha + (1 - 2*alpha)*shower
    shower = np.ma.log(x/(1-x)).filled(0)
    if logE:        
        return np.log10(e), shower, energy
    else:
        return e, shower, energy

class CaloDataset(data.Dataset):
    def __init__(self, cond_energy, voxel, energy, transform=None):
        self.cond_energy = torch.from_numpy(cond_energy).double()
        self.voxel = torch.from_numpy(voxel).double()
        self.energy = torch.from_numpy(energy).double()
        self.transform = transform
        
    def __getitem__(self, index):
        E = self.cond_energy[index]
        E_layers = self.energy[index]
        shower = self.voxel[index]
        
        return {'E': E, 'E0': E_layers[0].squeeze(), 'E1': E_layers[1].squeeze(), 
                'E2': E_layers[2].squeeze(), 'E3': E_layers[3].squeeze(), 
                'E4': E_layers[4].squeeze(), 'E5': E_layers[5].squeeze(), 
                'E6': E_layers[6].squeeze(), 'E7': E_layers[7].squeeze(), 
                'E8': E_layers[8].squeeze(), 'shower': shower}
    
    def __len__(self):
        return self.energy.shape[0]
    
def CaloFFJORD_prep(file_path,nevts=-1):    
    cond_energy, voxel, energy = DataLoaderDataset2(file_path,nevts)
    
    samples = CaloDataset(cond_energy, voxel, energy)
    nevts=energy.shape[0]
    frac = 0.8 #Fraction of events used for training
    train, test = data.random_split(samples,[int(frac*nevts), nevts - int((frac)*nevts)])
    return int(frac*nevts), nevts - int((frac)*nevts), train, test

if __name__ == "__main__":
    file_path = '/pscratch/sd/v/vmikuni/SGM/gamma.hdf5'
    energy, energy_layer, energy_voxel = DataLoader(file_path,1000)
    print(energy.shape, energy_layer.shape, energy_voxel.shape)
