""" Main script to run CaloFlow.
    code based on https://github.com/bayesiains/nflows and https://arxiv.org/pdf/1906.04032.pdf

    the current setup supports running in different modes:

    - train one flow for all layers combined, no additional conditioning
    - train one flow per layer and condition on previous layers (with varying options)
    - use one flow to learn p(E_i|E_tot) and then train one flow per layer on normalized samples
    - use one flow to learn p(E_i|E_tot) and then train a single flow on normalized samples

    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

######################################   Imports   #################################################
import argparse
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from nflows import transforms, distributions, flows

import preprocessing
import plot_calo_2
from data import get_dataloader
from data import save_samples_to_file
#from base_dist import ConditionalDiagonalHalfNormal

from scipy.stats import loguniform #log-uniform sampling

torch.set_default_dtype(torch.float64)

#####################################   Parser setup   #############################################
parser = argparse.ArgumentParser()

# usage modes
parser.add_argument('--mode', default='single_recursive',
                    help='Which global setup to use: "single", "three", '+\
                    '"single_recursive", or "three_recursive"')
parser.add_argument('--train', action='store_true', help='train a flow')
parser.add_argument('--generate', action='store_true', help='generate from a trained flow and plot')
parser.add_argument('--evaluate', action='store_true', help='evaluate LL of a trained flow')
parser.add_argument('--generate_to_file', action='store_true',
                    help='generate from a trained flow and save to file')
parser.add_argument('--save_only_weights', action='store_true',
                    help='Loads full model file (incl. optimizer) and saves only weights')


parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

parser.add_argument('--output_dir', default='./results', help='Where to store the output')
parser.add_argument('--results_file', default='results.txt',
                    help='Filename where to store settings and test results.')
parser.add_argument('--restore_file', type=str, default=None, help='Model file to restore.')
parser.add_argument('--data_dir', default='/media/claudius/8491-9E93/ML_sources/CaloGAN',
                    help='Where to find the training dataset')

# CALO specific
parser.add_argument('--with_noise', action='store_true',
                    help='Add 1e-8 noise (w.r.t. 100 GeV) to dataset to avoid voxel with 0 energy')
parser.add_argument('--particle_type', '-p',
                    help='Which particle to shower, "gamma", "eplus", or "piplus"')
parser.add_argument('--num_layer', default=8, type=int,
                    help='How many Calolayers are trained')
parser.add_argument('--energy_encoding', default='logdirect',
                    help='How the energy conditional is given to the NN: "direct", '+\
                    '"logdirect", "one_hot", or "one_blob"')
parser.add_argument('--energy_encoding_bins', default=100,
                    help='Number of bins in one_hot or one_blob energy encoding')
parser.add_argument('--layer_condition', default='energy', type=str,
                    help='How the condition is given into the network(s). '+\
                    'Must be one of ["None", "energy", "full", "NN"] for --mode=three '+\
                    'or ["NN"] for --mode=single.')
parser.add_argument('--normed', action='store_true',
                    help='Normalize voxels to total Energy per event')
parser.add_argument('--threshold', type=float, default=1e-4,
                    help='Threshold in MeV below which voxel energies are set to 0. in plots.')
parser.add_argument('--post_process', action='store_true',
                    help='Normalize sampled events to have E_tot (sampled from E_true) ')

# MAF parameters
parser.add_argument('--n_blocks', type=str, default='8',
                    help='Total number of blocks to stack in a model (MADE in MAF).')
parser.add_argument('--hidden_size', type=str, default='378',
                    help='Hidden layer size for each MADE block in an MAF.')
parser.add_argument('--hidden_size_multiplier', type=int, default=None,
                    help='Hidden layer size for each MADE block in an MAF'+\
                    ' is given by the dimension times this factor.')
parser.add_argument('--n_hidden', type=int, default=1,
                    help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu',
                    help='What activation function of torch.nn.functional to use in the MADEs.')
parser.add_argument('--batch_norm', action='store_true', default=False,
                    help='Use batch normalization')
parser.add_argument('--n_bins', type=int, default=8,
                    help='Number of bins if piecewise transforms are used')
parser.add_argument('--use_residual', action='store_true', default=False,
                    help='Use residual layers in the NNs')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.05,
                    help='dropout probability')
parser.add_argument('--tail_bound', type=float, default=14., help='Domain of the RQS')
parser.add_argument('--cond_base', action='store_true', default=False,
                    help='Use Gaussians conditioned on energy as base distribution.')
parser.add_argument('--init_id', action='store_true',
                    help='Initialize Flow to be identity transform')

# training params
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4, help='Initial Learning rate.')
parser.add_argument('--log_interval', type=int, default=175,
                    help='How often to show loss statistics and save samples.')


#######################################   helper functions   #######################################

# used in transformation between energy and logit space:
# (should match the ALPHA in data.py)
ALPHA = 1e-6

def logit(x):
    """ returns logit of input """
    return torch.log(x / (1.0 - x))

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((torch.sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clamp_(clamp_low, clamp_high)

def one_hot(values, num_bins):
    """ one-hot encoding of values into num_bins """
    # values are energies in [0, 1], need to be converted to integers in [0, num_bins-1]
    values *= num_bins
    values = values.type(torch.long)
    ret = F.one_hot(values, num_bins)
    return ret.squeeze().double()

def one_blob(values, num_bins):
    """ one-blob encoding of values into num_bins, cf sec. 4.3 of 1808.03856 """
    # torch.tile() not yet in stable release, use numpy instead
    values = values.cpu().numpy()[..., np.newaxis]
    y = np.tile(((0.5/num_bins) + np.arange(0., 1., step=1./num_bins)), values.shape)
    res = np.exp(((-num_bins*num_bins)/2.)
                 * (y-values)**2)
    res = np.reshape(res, (-1, values.shape[-1]*num_bins))
    return torch.tensor(res)

def remove_nans(tensor):
    """removes elements in the given batch that contain nans
       returns the new tensor and the number of removed elements"""
    tensor_flat = tensor.flatten(start_dim=1)
    good_entries = torch.all(tensor_flat == tensor_flat, axis=1)
    res_flat = tensor_flat[good_entries]
    tensor_shape = list(tensor.size())
    tensor_shape[0] = -1
    res = res_flat.reshape(tensor_shape)
    return res, len(tensor) - len(res)

def transform_to_energy(sample, args, scaling=0., Ehat=None):
    """ transforms samples from logit space to energy space, possibly applying a scaling
        with Etot (scaling) and/or Ehat
    """
    sample = ((torch.sigmoid(sample) - ALPHA) / (1. - 2.*ALPHA))
    if args.post_process and ('recursive' not in args.mode):
        sample = sample / sample.abs().sum(dim=(-1), keepdims=True)
        sample = sample.to('cpu')*Ehat
        scaling = scaling.to('cpu')

    if 'recursive' in args.mode:
        sample0, sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8 = torch.split(sample, args.dim_split, dim=-1)
        sample0 = (sample0 / sample0.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 0].reshape(-1, 1, 1)
        sample1 = (sample1 / sample1.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 1].reshape(-1, 1, 1)
        sample2 = (sample2 / sample2.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 2].reshape(-1, 1, 1)
        sample3 = (sample3 / sample3.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 3].reshape(-1, 1, 1)
        sample4 = (sample4 / sample4.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 4].reshape(-1, 1, 1)
        sample5 = (sample5 / sample5.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 5].reshape(-1, 1, 1)
        sample6 = (sample6 / sample6.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 6].reshape(-1, 1, 1)
        sample7 = (sample7 / sample7.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 7].reshape(-1, 1, 1)
        sample8 = (sample8 / sample8.abs().sum(dim=(-1), keepdims=True))\
            * scaling[:, 8].reshape(-1, 1, 1)
        sample = torch.cat((sample0, sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8), 2)
        sample = sample*1e3
    elif args.normed:
        sample = sample*scaling*1e3
    else:
        sample = sample*1e3
    return sample

def plot_sample(plot_layer_id, sample, args, step=None, n_col=10):
    """ plots event display for given sample """
    sample = sample.view(-1, *args.input_dims[str(plot_layer_id)])

    filename = 'generated_samples_layer_'+str(plot_layer_id) +\
                (step is not None)*'_epoch_{}'.format(step) +\
                '_'+str(args.threshold)+'.png'

    plot_calo_2.plot_calo_batch(sample, os.path.join(args.output_dir, filename),
                              plot_layer_id, ncol=n_col, lower_threshold=args.threshold)

def plot_all(samples, args, step=None, used_energies=0.):
    """ plots summary plots for large sample """
    sample_list = [*torch.split(samples, args.dim_split, dim=-1)]
    for plot_layer_id, sample in enumerate(sample_list):
        sample = sample.view(-1, *args.input_dims[str(plot_layer_id)])
        sample, num_bad = remove_nans(sample)

        print("Having {} bad samples of {}".format(num_bad, len(samples)))
        if num_bad == 0:
            print(sample.to('cpu').detach().numpy().min(),
                  sample.to('cpu').detach().numpy().max())


            filename = 'generated_samples_E_layer_'+str(plot_layer_id) +\
                (step is not None)*'_epoch_{}'.format(step) + '_'+str(args.threshold)+ '.png'
            plot_calo_2.plot_layer_energy(sample,
                                        os.path.join(args.output_dir, filename),
                                        plot_layer_id, plot_ref=args.particle_type,
                                        plot_GAN=None,
                                        epoch_nr=step, lower_threshold=args.threshold)
            
            filename = 'e_ratio_layer_'+str(plot_layer_id) +\
                (step is not None)*'_epoch_{}'.format(step) + '_'+str(args.threshold)+ '.png'
            plot_calo_2.plot_layer_E_ratio(sample,
                                         os.path.join(args.output_dir, filename),
                                         plot_layer_id, plot_ref=args.particle_type,
                                         plot_GAN=None,
                                         epoch_nr=step, lower_threshold=args.threshold)
        else:
            print("Skipping plotting due to bad sample")

    if args.num_layer == 8: 

        filename = 'generated_samples_E_total' +\
            (step is not None)*'_epoch_{}'.format(step) + '_'+str(args.threshold)+ '.png'
        plot_calo_2.plot_total_energy(samples,
                                    os.path.join(args.output_dir, filename),
                                    plot_ref=args.particle_type,
                                    plot_GAN=args.particle_type,
                                    epoch_nr=step,
                                    lower_threshold=args.threshold)
        



def split_and_concat(generate_fun, batch_size, model, args, num_pts, energies, rec_model=None):
    """ generates events in batches of size batch_size, if needed """
    starting_time = time.time()
    energy_split = energies.split(batch_size)
    ret = []
    for iteration, energy_entry in enumerate(energy_split):
        if 'recursive' in args.mode:
            ret.append(generate_fun(model, args, num_pts, energy_entry, rec_model).to('cpu'))
        else:
            ret.append(generate_fun(model, args, num_pts, energy_entry).to('cpu'))
        print("Generated {}%".format((iteration+1.)*100. / len(energy_split)), end='\r')
    ending_time = time.time()
    total_time = ending_time - starting_time
    time_string = "Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+\
        " This means {:.2f} ms per event."
    print(time_string.format(int(total_time//60), total_time%60, num_pts*len(energies),
                             len(energy_split), total_time*1e3 / (num_pts*len(energies))))
    print(time_string.format(int(total_time//60), total_time%60, num_pts*len(energies),
                             len(energy_split), total_time*1e3 / (num_pts*len(energies))),
          file=open(args.results_file, 'a'))
    return torch.cat(ret)

@torch.no_grad()
def generate(model, args, energies=None, n_col=10, step=None, include_average=False,
             rec_model=None):
    """ generate samples from the trained model and plots histograms"""
    if energies is None:
        energies = torch.arange(0.01, 1.01, 0.01)
        #print("Are we ending up here, by any chance?")
        energies[-1] -= 1e-6
    if args.post_process:
        Ehat = get_Ehat(energies, args)
    else:
        Ehat = None

    scaling = torch.reshape(energies, (-1, 1, 1)).to(args.device)

    if args.mode == 'single_recursive':
        samples = generate_single_with_rec(model, args, 1, energies, rec_model)
    else:
        samples = generate_single_with_rec(model, args, 1, energies, rec_model)

    sample_list = [*torch.split(samples, args.dim_split, dim=-1)]

    #for plot_layer_id, sample in enumerate(sample_list):
        #plot_sample(plot_layer_id, sample, args, step=step, n_col=n_col)
    #change to 10k
    if include_average and ((step % 10 == 0) or step == args.n_epochs):
        num_pts = 10000 if (step == args.n_epochs) else 10000
        energies = torch.from_numpy(loguniform.rvs(1e-3, 1e0, size=num_pts))
        #energies = 0.999*torch.rand((num_pts,)) + 0.001
        scaling = torch.reshape(energies, (-1, 1, 1)).to(args.device)
        #print("This is the scaling:")
        print(scaling)
        if args.post_process:
            #print("Uh-oh, E hat is being used and something could be wrong!")
            Ehat = get_Ehat(energies, args)
        else:
            Ehat = None

        if args.mode == 'single_recursive':
            samples = split_and_concat(generate_single_with_rec, 10000, model, args, 1, energies,
                                       rec_model)
        else:
            samples = split_and_concat(generate_single_with_rec, 10000, model, args, 1, energies,
                                       rec_model)
        print(samples)
        plot_all(samples, args, step=step, used_energies=scaling)
        if args.generate_to_file:
            filename = os.path.join(args.output_dir, 'CaloFlow_'+args.particle_type+'.hdf5')
            save_samples_to_file(samples, energies, filename, args.threshold)


def generate_to_file(model, args, num_events=100000, energies=None, rec_model=None):
    """ generates samples from the trained model and saves them to file """
    if energies is None:
        energies = 0.99*torch.rand((num_events,)) + 0.01
    scaling = torch.reshape(energies, (-1, 1, 1)).to(args.device)
    if args.post_process:
        Ehat = get_Ehat(energies, args)
    else:
        Ehat = None

    if args.mode == 'single':
        samples = split_and_concat(generate_single, 10000, model, args, 1, energies)
    elif args.mode == 'single_recursive':
        samples = split_and_concat(generate_single_with_rec, 10000, model, args, 1, energies,
                                   rec_model)
    elif args.mode == 'three':
        samples = split_and_concat(generate_three, 25000, model, args, 1, energies)
    else:
        samples = split_and_concat(generate_with_rec, 25000, model, args, 1, energies,
                                   rec_model)
    if args.mode not in ['single_recursive', 'three_recursive']:
        samples = transform_to_energy(samples, args, scaling=scaling, Ehat=Ehat)
    filename = os.path.join(args.output_dir, 'CaloFlow_'+args.particle_type+'.hdf5')
    save_samples_to_file(samples, energies, filename, args.threshold)

def train_and_evaluate(model, train_loader, test_loader, optimizer, args, rec_model=None):
    """ As the name says, train the flow and evaluate along the way """
    best_eval_logprob = float('-inf')
    if 'recursive' in args.mode:
        milestones = [50]
    else:
        milestones = [15, 40, 70, 100, 150]
    if args.mode in ['single', 'single_recursive']:
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=milestones,
                                                           gamma=0.5,
                                                           verbose=True)
    else:
        lr_schedule = []
        for optim in optimizer:
            lr_schedule.append(torch.optim.lr_scheduler.MultiStepLR(optim,
                                                                    milestones=milestones,
                                                                    gamma=0.5,
                                                                    verbose=True))
    for i in range(args.n_epochs):
        train(model, train_loader, optimizer, i, args)
        with torch.no_grad():
            eval_logprob, _ = evaluate(model, test_loader, i, args)
            args.test_loss.append(-eval_logprob.to('cpu').numpy())
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            save_all(model, optimizer, args)
        with torch.no_grad():
            generate(model, args, step=i+1, include_average=True, rec_model=rec_model)
        if args.mode in ['single', 'single_recursive']:
            lr_schedule.step()
        else:
            for schedule in lr_schedule:
                schedule.step()
        plot_calo_2.plot_loss(args.train_loss, args.test_loss,
                            [os.path.join(args.output_dir, 'loss.png'),
                             os.path.join(args.output_dir, 'test_loss.npy'),
                             os.path.join(args.output_dir, 'train_loss.npy')])

def train(model, dataloader, optimizer, epoch, args):
    """ train the flow one epoch """
    if args.mode == 'single_recursive':
        train_single_with_rec(model, dataloader, optimizer, epoch, args)
    else:
        train_single_with_rec(model, dataloader, optimizer, epoch, args)

@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    """Evaluate the model, i.e find the mean log_prob of the test set
       Energy is taken to be the energy of the image, so no
       marginalization is performed.
    """
    if args.mode == 'single_recursive':
        return evaluate_single_with_rec(model, dataloader, epoch, args)
    else:
        return evaluate_single_with_rec(model, dataloader, epoch, args)

def save_all(model, optimizer, args):
    """ saves the model and the optimizer to file """
    if args.mode in ['single', 'single_recursive']:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(args.output_dir, 'model_checkpoint.pt'))
    else:
        model_dicts = {}
        optim_dicts = {}
        for num_layer in range(args.num_layer+1):
            model_dicts[num_layer] = model[num_layer].state_dict()
            optim_dicts[num_layer] = optimizer[num_layer].state_dict()
        torch.save({'model_state_dict': model_dicts,
                    'optimizer_state_dict': optim_dicts},
                   os.path.join(args.output_dir, 'model_checkpoint.pt'))

def save_weights(model, args):
    """ saves the model to file """
    if args.mode in ['single', 'single_recursive']:
        torch.save({'model_state_dict': model.state_dict()},
                   os.path.join(args.output_dir, 'weight_checkpoint.pt'))
    else:
        model_dicts = {}
        for num_layer in range(args.num_layer+1):
            model_dicts[num_layer] = model[num_layer].state_dict()
        torch.save({'model_state_dict': model_dicts},
                   os.path.join(args.output_dir, 'weight_checkpoint.pt'))

def load_all(model, optimizer, args):
    """ loads the model and optimizer from file """
    filename = args.restore_file if args.restore_file is not None else 'model_checkpoint.pt'
    #checkpoint = torch.load(os.path.join(args.output_dir, filename))
    checkpoint = torch.load(filename)
    if args.mode in ['single', 'single_recursive']:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(args.device)
        model.eval()
    else:
        for num_layer in range(args.num_layer+1):
            model[num_layer].load_state_dict(checkpoint['model_state_dict'][num_layer])
            optimizer[num_layer].load_state_dict(checkpoint['optimizer_state_dict'][num_layer])
            model[num_layer].to(args.device)
            model[num_layer].eval()

def load_weights(model, args):
    """ loads the model from file """
    filename = args.restore_file if args.restore_file is not None else 'weight_checkpoint.pt'
    checkpoint = torch.load(filename)
    if args.mode in ['single', 'single_recursive']:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        model.eval()
    else:
        for num_layer in range(args.num_layer+1):
            model[num_layer].load_state_dict(checkpoint['model_state_dict'][num_layer])
            model[num_layer].to(args.device)
            model[num_layer].eval()

def save_rec_flow(rec_model, args):
    """saves flow that learns energies recursively """
    torch.save({'model_state_dict': rec_model.state_dict()},
               os.path.join('./rec_energy_flow_v5/', args.particle_type+'.pt'))

def load_rec_flow(rec_model, args):
    """ loads flow that learns energies recursively """
    checkpoint = torch.load(os.path.join('./rec_energy_flow_v5/', args.particle_type+'.pt'))
    rec_model.load_state_dict(checkpoint['model_state_dict'])
    rec_model.to(args.device)
    rec_model.eval()

def get_Ehat(energies, args):
    """ sample Ehat (sum of recorded energies) for set of Etot (target total energies)
        use previously computed histograms based on training data for distribution
    """
    energies = energies.to('cpu').numpy()*1e2
    sampling_hists_file = pd.read_hdf('energy_sampling.hdf')
    sampling_hists = sampling_hists_file.loc[args.particle_type].to_numpy()
    num_bins = len(sampling_hists)
    bins_ext = np.linspace(0., 100., num_bins+1)
    bins_int = np.linspace(0., 1., num_bins+1)
    energy_which_histo = np.searchsorted(bins_ext, energies)-1
    ret = sample_from_histograms(sampling_hists[energy_which_histo], bins_int)
    return ret.reshape(-1, 1, 1)

def sample_from_histograms(histos, bins):
    """ Sample random numbers according to given cdfs """
    numpts = len(histos)
    rnd = np.random.rand(2*numpts)
    which_bins = histo_searchsorted(histos, rnd[:numpts])
    ret = bins[which_bins] + (rnd[numpts:]/len(bins))
    return ret

def histo_searchsorted(a, b):
    """ like numpy searchsorted, but a can be 2-dim and b 1-dim, assumes max(a) = 1 """
    m, n = a.shape
    r = np.arange(m)[:, None]
    p = np.searchsorted((a+r).ravel(), (b[..., np.newaxis]+r).ravel()).reshape(m, -1)
    return (p-n*(np.arange(m)[:, None])).reshape(b.shape)

def trafo_to_unit_space(energy_array, max_deposit=2.3):
    """ transforms energy array to be in [0, 1] """
    num_dim = len(energy_array[0])-2
    ret = [(torch.sum(energy_array[:, :-1], dim=1)/(energy_array[:, -1] * max_deposit)).unsqueeze(1)]
    for n in range(num_dim):
        ret.append((energy_array[:, n]/energy_array[:, n:-1].sum(dim=1)).unsqueeze(1))
    #print("Maximum in unit space")
    #print(torch.max(torch.nan_to_num(torch.cat(ret, 1).clamp_(0., 1.))))
    return torch.nan_to_num(torch.cat(ret, 1).clamp_(0., 1.))

def trafo_to_energy_space(unit_array, etot_array, max_deposit=2.3):
    """ transforms unit array to be back in energy space """
    assert len(unit_array) == len(etot_array)
    num_dim = len(unit_array[0])
    unit_array = torch.cat((unit_array, torch.ones(size=(len(unit_array), 1))), 1)
    ret = [torch.zeros_like(etot_array)]
    ehat_array = unit_array[:, 0] * (etot_array * max_deposit)
    for n in range(num_dim):
        ret.append(unit_array[:, n+1]*(ehat_array-torch.cat(ret).view(
            n+1, -1).transpose(0, 1).sum(dim=1)))
    ret.append(etot_array)
    #print("Maximum back to energy space")
    #print(torch.max(torch.cat(ret).view(num_dim+2, -1)[1:].transpose(0, 1)))
    return torch.cat(ret).view(num_dim+2, -1)[1:].transpose(0, 1)

################################# auxilliary NNs and classes #######################################

class BaseContext(torch.nn.Module):
    """ Small NN to map energy input to mean and width of base_gaussians"""
    def __init__(self, context_size, dimensionality):
        """ context_size: length of context vector
            dimensionality: number of dimensions of base dist.
        """
        super(BaseContext, self).__init__()
        self.layer1 = torch.nn.Linear(context_size, dimensionality)
        self.layer2 = torch.nn.Linear(dimensionality, dimensionality)
        self.output = torch.nn.Linear(dimensionality, 2*dimensionality)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = self.output(x)
        return out

class RandomPermutationLayer(transforms.Permutation):
    """ Permutes elements with random, but fixed permutation. Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be permuted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.random.permutation(features_entry)
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)

class InversionLayer(transforms.Permutation):
    """ Inverts the order of the elements in each layer.  Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be inverted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.arange(features_entry)[::-1]
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)

################## train and evaluation functions for recursive layer three flow ###################

def train_single_with_rec(model, dataloader, optimizer, epoch, args):
    """ train recursive single flow one step """
    model.train()
    for i, data in enumerate(dataloader):
        E = data['E']
            
        E0 = data['E0']
        E1 = data['E1']
        E2 = data['E2']
        E3 = data['E3']
        E4 = data['E4']
        E5 = data['E5']
        E6 = data['E6']
        E7 = data['E7']
        E8 = data['E8']
        
        x0 = data['shower'][:,0*64:1*64]
        x1 = data['shower'][:,1*64:2*64]
        x2 = data['shower'][:,2*64:3*64]
        x3 = data['shower'][:,3*64:4*64]
        x4 = data['shower'][:,4*64:5*64]
        x5 = data['shower'][:,5*64:6*64]
        x6 = data['shower'][:,6*64:7*64]
        x7 = data['shower'][:,7*64:8*64]
        x8 = data['shower'][:,8*64:9*64]

        energy_dists = trafo_to_unit_space(torch.cat((E0.unsqueeze(1), #might be an issue
                                                      E1.unsqueeze(1),
                                                      E2.unsqueeze(1),
                                                      E3.unsqueeze(1), 
                                                      E4.unsqueeze(1),
                                                      E5.unsqueeze(1),
                                                      E6.unsqueeze(1), 
                                                      E7.unsqueeze(1),
                                                      E8.unsqueeze(1),
                                                      E), 1))
        if args.energy_encoding == 'logdirect': #units -7 to 2 vs -9 to 0 (removing + 2), 1e-8 to 1e-9
            energy = torch.log10(E*(10.**(3/2)))
            E0 = torch.log10(E0.unsqueeze(-1)+1e-9) + 3.
            E1 = torch.log10(E1.unsqueeze(-1)+1e-9) + 3.
            E2 = torch.log10(E2.unsqueeze(-1)+1e-9) + 3.
            E3 = torch.log10(E3.unsqueeze(-1)+1e-9) + 3.
            E4 = torch.log10(E4.unsqueeze(-1)+1e-9) + 3.
            E5 = torch.log10(E5.unsqueeze(-1)+1e-9) + 3.
            E6 = torch.log10(E6.unsqueeze(-1)+1e-9) + 3.
            E7 = torch.log10(E7.unsqueeze(-1)+1e-9) + 3.
            E8 = torch.log10(E8.unsqueeze(-1)+1e-9) + 3.

        #y = torch.cat((energy, energy_dists, E0, E1, E2), 1).to(args.device)
        y = torch.cat((energy, E0, E1, E2, E3, E4, E5, E6, E7, E8), 1).to(args.device)

        if args.num_layer == 8:
            layer0 = x0.view(x0.shape[0], -1).to(args.device)
            layer1 = x1.view(x1.shape[0], -1).to(args.device)
            layer2 = x2.view(x2.shape[0], -1).to(args.device)
            layer3 = x3.view(x3.shape[0], -1).to(args.device)
            layer4 = x4.view(x4.shape[0], -1).to(args.device)
            layer5 = x5.view(x5.shape[0], -1).to(args.device)
            layer6 = x6.view(x6.shape[0], -1).to(args.device)
            layer7 = x7.view(x7.shape[0], -1).to(args.device)
            layer8 = x8.view(x8.shape[0], -1).to(args.device)
            x = torch.cat((layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8), 1)

        loss = - model.log_prob(x, y).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.train_loss.append(loss.tolist())

        if i % args.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, args.n_epochs, i, len(dataloader), loss.item()))
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, args.n_epochs, i, len(dataloader), loss.item()),
                  file=open(args.results_file, 'a'))

@torch.no_grad()
def evaluate_single_with_rec(model, dataloader, epoch, args):
    """Evaluate the single flow with energy distribution flow, i.e find the mean log_prob
       of the test set. Energy is taken to be the energy of the event, so no
       marginalization is performed.
    """
    model.eval()
    loglike = []
    for i, data in enumerate(dataloader):
        E = data['E']
            
        E0 = data['E0']
        E1 = data['E1']
        E2 = data['E2']
        E3 = data['E3']
        E4 = data['E4']
        E5 = data['E5']
        E6 = data['E6']
        E7 = data['E7']
        E8 = data['E8']
        
        x0 = data['shower'][:,0*64:1*64]
        x1 = data['shower'][:,1*64:2*64]
        x2 = data['shower'][:,2*64:3*64]
        x3 = data['shower'][:,3*64:4*64]
        x4 = data['shower'][:,4*64:5*64]
        x5 = data['shower'][:,5*64:6*64]
        x6 = data['shower'][:,6*64:7*64]
        x7 = data['shower'][:,7*64:8*64]
        x8 = data['shower'][:,8*64:9*64]

        energy_dists = trafo_to_unit_space(torch.cat((E0.unsqueeze(1), #might be an issue
                                                      E1.unsqueeze(1),
                                                      E2.unsqueeze(1),
                                                      E3.unsqueeze(1), 
                                                      E4.unsqueeze(1),
                                                      E5.unsqueeze(1),
                                                      E6.unsqueeze(1), 
                                                      E7.unsqueeze(1),
                                                      E8.unsqueeze(1),
                                                      E), 1))
        if args.energy_encoding == 'logdirect':
            energy = torch.log10(E*(10.**(3/2)))
            E0 = torch.log10(E0.unsqueeze(-1)+1e-9) + 3.
            E1 = torch.log10(E1.unsqueeze(-1)+1e-9) + 3.
            E2 = torch.log10(E2.unsqueeze(-1)+1e-9) + 3.
            E3 = torch.log10(E3.unsqueeze(-1)+1e-9) + 3.
            E4 = torch.log10(E4.unsqueeze(-1)+1e-9) + 3.
            E5 = torch.log10(E5.unsqueeze(-1)+1e-9) + 3.
            E6 = torch.log10(E6.unsqueeze(-1)+1e-9) + 3.
            E7 = torch.log10(E7.unsqueeze(-1)+1e-9) + 3.
            E8 = torch.log10(E8.unsqueeze(-1)+1e-9) + 3.

        #y = torch.cat((energy, energy_dists, E0, E1, E2), 1).to(args.device)
        y = torch.cat((energy, E0, E1, E2, E3, E4, E5, E6, E7, E8), 1).to(args.device)

        if args.num_layer == 8:
            layer0 = x0.view(x0.shape[0], -1).to(args.device)
            layer1 = x1.view(x1.shape[0], -1).to(args.device)
            layer2 = x2.view(x2.shape[0], -1).to(args.device)
            layer3 = x3.view(x3.shape[0], -1).to(args.device)
            layer4 = x4.view(x4.shape[0], -1).to(args.device)
            layer5 = x5.view(x5.shape[0], -1).to(args.device)
            layer6 = x6.view(x6.shape[0], -1).to(args.device)
            layer7 = x7.view(x7.shape[0], -1).to(args.device)
            layer8 = x8.view(x8.shape[0], -1).to(args.device)
            x = torch.cat((layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8), 1)

        loglike.append(model.log_prob(x, y))

    logprobs = torch.cat(loglike, dim=0).to(args.device)

    logprob_mean = logprobs.mean(0)
    logprob_std = logprobs.var(0).sqrt() / np.sqrt(len(dataloader.dataset))

    output = 'Evaluate ' + (epoch is not None)*'(epoch {}) -- '.format(epoch+1) +\
        'logp(x, at E(x)) = {:.3f} +/- {:.3f}'

    print(output.format(logprob_mean, logprob_std))
    print(output.format(logprob_mean, logprob_std), file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std

@torch.no_grad()
def generate_single_with_rec(model_list, args, num_pts, energies, rec_model):
    """ Generate Samples from single flow with energy flow """
    model.eval()

    energy_dist_unit = sample_rec_flow(rec_model, num_pts, args, energies).to('cpu')
    energy_dist = trafo_to_energy_space(energy_dist_unit, energies)
    if args.energy_encoding == 'logdirect':
        energies = torch.log10(energies*(10.**(3/2))).unsqueeze(-1)
        E0 = torch.log10(energy_dist[:, 0].unsqueeze(-1)+1e-9) + 3.
        E1 = torch.log10(energy_dist[:, 1].unsqueeze(-1)+1e-9) + 3.
        E2 = torch.log10(energy_dist[:, 2].unsqueeze(-1)+1e-9) + 3.
        E3 = torch.log10(energy_dist[:, 3].unsqueeze(-1)+1e-9) + 3.
        E4 = torch.log10(energy_dist[:, 4].unsqueeze(-1)+1e-9) + 3.
        E5 = torch.log10(energy_dist[:, 5].unsqueeze(-1)+1e-9) + 3.
        E6 = torch.log10(energy_dist[:, 6].unsqueeze(-1)+1e-9) + 3.
        E7 = torch.log10(energy_dist[:, 7].unsqueeze(-1)+1e-9) + 3.
        E8 = torch.log10(energy_dist[:, 8].unsqueeze(-1)+1e-9) + 3.

    #y = torch.cat((energies, energy_dist_unit, E0, E1, E2), 1).to(args.device)
    y = torch.cat((energies, E0, E1, E2, E3, E4, E5, E6, E7, E8), 1).to(args.device)
    
    #Verifying the ranges
    #print("log(E) max for layers")
    #print(torch.max(y[:,1:]))
    
    #print("log(E) min for layers")
    #print(torch.min(y[:,1:]))
    
    #print("log(E) max for total E")
    #print(torch.max(y[:,0]))
    
    #print("log(E) min for total E")
    #print(torch.min(y[:,0]))
    
    samples = model.sample(num_pts, y)

    samples = transform_to_energy(samples, args, scaling=energy_dist.to(args.device))

    return samples

################## train and evaluation functions for recursive flow ###############################

def train_rec_flow(rec_model, train_data, test_data, optim, args):
    """ trains the flow that learns the energy distributions """
    best_eval_logprob_rec = float('-inf')

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                       milestones=[5, 10, 30, 40],
                                                       gamma=0.5,
                                                       verbose=True)

    num_epochs = 100
    for epoch in range(num_epochs):
        rec_model.train()
        for i, data in enumerate(train_data):
            E = data['E']
            
            E0 = data['E0']
            E1 = data['E1']
            E2 = data['E2']
            E3 = data['E3']
            E4 = data['E4']
            E5 = data['E5']
            E6 = data['E6']
            E7 = data['E7']
            E8 = data['E8']
            
            #print(E0)
            
            #y = one_blob(E, 10).to(args.device)
            y = torch.log10(E*(10.**(3/2))).to(args.device)

            x = trafo_to_unit_space(torch.cat((E0.unsqueeze(1), #might be an issue
                                               E1.unsqueeze(1),
                                               E2.unsqueeze(1),
                                               E3.unsqueeze(1), 
                                               E4.unsqueeze(1),
                                               E5.unsqueeze(1),
                                               E6.unsqueeze(1), 
                                               E7.unsqueeze(1),
                                               E8.unsqueeze(1),
                                               E), 1)).to(args.device)
            #print(x)
            x = logit_trafo(x)
            loss = - rec_model.log_prob(x, y).mean(0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % args.log_interval == 0:
                print('Recursive Flow: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, i, len(train_data), loss.item()))
                print('Recursive Flow: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, i, len(train_data), loss.item()),
                      file=open(args.results_file, 'a'))

        with torch.no_grad():
            rec_model.eval()
            loglike = []
            for data in test_data:
                E0 = data['E0']
                E1 = data['E1']
                E2 = data['E2']
                E3 = data['E3']
                E4 = data['E4']
                E5 = data['E5']
                E6 = data['E6']
                E7 = data['E7']
                E8 = data['E8']
                E  = data['E']
                #y = one_blob(E, 10).to(args.device)
                y = torch.log10(E*(10.**(3/2))).to(args.device)
                x = trafo_to_unit_space(torch.cat((E0.unsqueeze(1), #might be an issue
                                                   E1.unsqueeze(1),
                                                   E2.unsqueeze(1),
                                                   E3.unsqueeze(1), 
                                                   E4.unsqueeze(1),
                                                   E5.unsqueeze(1),
                                                   E6.unsqueeze(1), 
                                                   E7.unsqueeze(1),
                                                   E8.unsqueeze(1),
                                                   E), 1)).to(args.device)

                x = logit_trafo(x)
                loglike.append(rec_model.log_prob(x, y))

            logprobs = torch.cat(loglike, dim=0).to(args.device)

            logprob_mean = logprobs.mean(0)
            logprob_std = logprobs.var(0).sqrt() / np.sqrt(len(test_data.dataset))

            output = 'Recursive Flow: Evaluate (epoch {}) -- '.format(epoch+1) +\
                'logp(x, at E(x)) = {:.3f} +/- {:.3f}'

            print(output.format(logprob_mean, logprob_std))
            print(output.format(logprob_mean, logprob_std), file=open(args.results_file, 'a'))
            eval_logprob_rec = logprob_mean
        lr_schedule.step()
        if eval_logprob_rec > best_eval_logprob_rec:
            best_eval_logprob_rec = eval_logprob_rec
            save_rec_flow(rec_model, args)

@torch.no_grad()
def sample_rec_flow(rec_model, num_pts, args, energies):
    """ samples layer energies for given total energy from rec flow """
    rec_model.eval()
    #context = one_blob(energies, 10).to(args.device)
    context = torch.log10(energies*(10.**(3/2))).to(args.device)
    samples = rec_model.sample(num_pts, context.unsqueeze(-1))
    samples = inverse_logit(samples.squeeze())
    return samples

####################################################################################################
#######################################   running the code   #######################################
####################################################################################################

if __name__ == '__main__':
    sample_e = torch.Tensor([[1, 0.3, 0.2, 0.3, 0.1, 0.001, 0.001, 0.001, 1.903]])
    print("Sample E")
    print(sample_e)
    print("Unit Space")
    traf_e = trafo_to_unit_space(sample_e)
    print(traf_e)
    print("There and Back")
    print(trafo_to_energy_space(traf_e, torch.Tensor([1.903])))
    
    args = parser.parse_args()

    # check if parsed arguments are valid

    assert args.num_layer in [8], (
        "Calorimeter has 8 layers")
    assert args.mode in ['single', 'three', 'single_recursive', 'three_recursive'], (
        "CaloFlow only supports running in modes 'single', 'three',"+\
        " 'single_recursive', or 'three_recursive'")
    assert (args.train or args.generate or args.evaluate or args.generate_to_file or \
        args.save_only_weights), (
        "Please specify at least one of --train, --generate, --evaluate, --generate_to_file")
    assert args.particle_type in ['gamma', 'eplus', 'piplus'], \
        'Particle type must be either gamma, eplus, or piplus!'
    assert args.layer_condition in ["None", "energy", "full", "NN"], (
        'layer_condition not in ["None", "energy", "full", "NN"]')

    if args.energy_encoding in ['direct', 'logdirect']:
        cond_label_size = 1
    elif args.energy_encoding in ['one_hot', 'one_blob']:
        cond_label_size = int(args.energy_encoding_bins)
    else:
        raise ValueError('energy_encoding not in ["direct", "logdirect", "one_hot", "one_blob"]')


    # check if output_dir exists and 'move' results file there
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.results_file = os.path.join(args.output_dir, args.results_file)
    print(args, file=open(args.results_file, 'a'))

    if "," in args.hidden_size:
        if args.mode in ['single', 'single_recursive']:
            raise ValueError("single-type modes only support a single hidden_size, no list")
        args_list = [int(item) for item in args.hidden_size.split(',')]
        args.hidden_size = args_list
        assert len(args.hidden_size) == args.num_layer+1
    else:
        if args.mode not in ['single', 'single_recursive']:
            args_list = (args.num_layer+1)*[int(args.hidden_size)]
            args.hidden_size = args_list
        else:
            args.hidden_size = int(args.hidden_size)

    if "," in args.n_blocks:
        if args.mode in ['single', 'single_recursive']:
            raise ValueError("single-type modes only support a single n_blocks, no list")
        args_list = [int(item) for item in args.n_blocks.split(',')]
        args.n_blocks = args_list
        assert len(args.n_blocks) == args.num_layer+1
    else:
        if args.mode not in ['single', 'single_recursive']:
            args_list = (args.num_layer+1)*[int(args.n_blocks)]
            args.n_blocks = args_list
        else:
            args.n_blocks = int(args.n_blocks)

    # setup device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))
    print("Using {}".format(args.device), file=open(args.results_file, 'a'))
    
    #filepath to caloflow data
    file_path = '/global/cfs/cdirs/m3929/SCRATCH/FCC/dataset_2_2_small.hdf5'
    ntrain, ntest, train_data, test_data = preprocessing.CaloFFJORD_prep(file_path)
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.device.type is 'cuda' else {}
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    print(ntrain)
    print(ntest)
    
    # get dataloaders
    # train_dataloader, test_dataloader = get_dataloader(args.particle_type,
                                                       #args.data_dir,
                                                       #full=False,
                                                       #apply_logit=True,
                                                       #device=args.device,
                                                       #batch_size=args.batch_size,
                                                       #with_noise=args.with_noise,
                                                       #normed=args.normed,
                                                       #normed_layer=
                                                       #('recursive' in args.mode))

    args.input_size = [64] * 9
    args.input_dims = {'0': (8, 8), '1': (8, 8), '2': (8, 8), 
                       '3': (8, 8), '4': (8, 8), '5': (8, 8),
                       '6': (8, 8), '7': (8, 8), '8': (8, 8)} 
    
    #train_dataloader.dataset.input_dims #honestly have no clue what this is

    flow_params_rec_energy = {'num_blocks': 2, #num of layers per block
                              'features': args.num_layer+1,
                              'context_features': 1, #10,
                              'hidden_features': 64,
                              'use_residual_blocks': False,
                              'use_batch_norm': False,
                              'dropout_probability': 0.,
                              'activation':getattr(F, args.activation_fn),
                              'random_mask': False,
                              'num_bins': 8,
                              'tails':'linear',
                              'tail_bound': 14,
                              'min_bin_width': 1e-6,
                              'min_bin_height': 1e-6,
                              'min_derivative': 1e-6}
    rec_flow_blocks = []
    for _ in range(6): #6 MADE blocks could be arbitrary here, will switch to 8
        rec_flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_rec_energy))
        rec_flow_blocks.append(transforms.RandomPermutation(args.num_layer+1))
    rec_flow_transform = transforms.CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = distributions.StandardNormal(shape=[args.num_layer+1])
    rec_flow = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution)

    rec_model = rec_flow.to(args.device)
    rec_optimizer = torch.optim.Adam(rec_model.parameters(), lr=1e-4)
    print(rec_model)
    print(rec_model, file=open(args.results_file, 'a'))

    total_parameters = sum(p.numel() for p in rec_model.parameters() if p.requires_grad)

    print("Recursive energy setup has {} parameters".format(int(total_parameters)))
    print("Recursive energy setup has {} parameters".format(int(total_parameters)),
          file=open(args.results_file, 'a'))

    if os.path.exists(os.path.join('./rec_energy_flow_v5/', args.particle_type+'.pt')):
        print("loading recursive energy flow")
        print("loading recursive energy flow", file=open(args.results_file, 'a'))
        load_rec_flow(rec_model, args)
    else:
        train_rec_flow(rec_model, train_dataloader, test_dataloader, rec_optimizer, args)
        print("loading recursive energy flow")
        print("loading recursive energy flow", file=open(args.results_file, 'a'))
        load_rec_flow(rec_model, args)

    # code training/evaluation/generation of three-rec

    # to plot losses:
    args.train_loss = []
    args.test_loss = []
    # to keep track of dimensionality in constructing the flows
    args.dim_sum = 0
    args.dim_split = []

    flow_params_RQS = {'num_blocks':args.n_hidden, # num of hidden layers per block
                       'use_residual_blocks':args.use_residual,
                       'use_batch_norm':args.batch_norm,
                       'dropout_probability':args.dropout_probability,
                       'activation':getattr(F, args.activation_fn),
                       'random_mask':False,
                       'num_bins':args.n_bins,
                       'tails':'linear',
                       'tail_bound':args.tail_bound,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}

    # setup flow
    if args.mode in ['single', 'single_recursive']:
        # setup single-type flows (plain or recursive)
        flow_blocks = []
        for layer_id in range(args.num_layer+1):
            current_dim = args.input_size[layer_id]
            args.dim_split.append(current_dim)
        for entry in args.dim_split:
            args.dim_sum += entry
        if args.mode == 'single_recursive':
            cond_label_size *= (2+args.num_layer)
            #cond_label_size += args.num_layer + 1
        for i in range(args.n_blocks):
            flow_blocks.append(
                transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    **flow_params_RQS,
                    features=args.dim_sum,
                    context_features=cond_label_size,
                    hidden_features=args.hidden_size
                ))
            if args.init_id:
                torch.nn.init.zeros_(flow_blocks[-1].autoregressive_net.final_layer.weight)
                torch.nn.init.constant_(flow_blocks[-1].autoregressive_net.final_layer.bias,
                                        np.log(np.exp(1 - 1e-6) - 1))

            if i%2 == 0:
                flow_blocks.append(InversionLayer(args.dim_split))
            else:
                flow_blocks.append(RandomPermutationLayer(args.dim_split))

        del flow_blocks[-1]
        flow_transform = transforms.CompositeTransform(flow_blocks)
        if args.cond_base:
            flow_base_distribution = distributions.ConditionalDiagonalNormal(
                shape=[args.dim_sum],
                context_encoder=BaseContext(
                    cond_label_size, args.dim_sum))
        else:
            flow_base_distribution = distributions.StandardNormal(shape=[args.dim_sum])
        flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

        model = flow.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        print(model)
        print(model, file=open(args.results_file, 'a'))

        total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total setup has {} parameters".format(int(total_parameters)))
    print("Total setup has {} parameters".format(int(total_parameters)),
          file=open(args.results_file, 'a'))


    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, args,
                           rec_model=rec_model if ('recursive' in args.mode) else None)

    if args.generate:
        #add energy parser, n_sample parser, write_to_file option
        #ref_energy, _, _ = preprocessing.DataLoaderDataset2(file_path, 10000)
        #ref_energies = torch.from_numpy(np.reshape(ref_energy,(ref_energy.shape[0], -1))).float()
        #true_energy, _ = preprocessing.ReverseNormChallenge(ref_energy, ref_energy_voxel)
        #ref_energies = torch.from_numpy(ref_energy).double().squeeze(1)
        #print(ref_energies.shape)
        #energies = ref_energies
        load_weights(model, args)
        generate(model, args, energies=None, step=args.n_epochs, include_average=True,
                 rec_model=rec_model if ('recursive' in args.mode) else None)

    if args.evaluate:
        load_weights(model, args)
        evaluate(model, test_dataloader, args.n_epochs, args)

    if args.generate_to_file and not args.generate:
        load_weights(model, args)
        # for nn plots
        #my_energies = torch.tensor(2000*[0.05, 0.1, 0.2, 0.5, 0.95])
        generate_to_file(model, args, num_events=100000, energies=None, #my_energies,
                         rec_model=rec_model if ('recursive' in args.mode) else None)

    if args.save_only_weights:
        load_all(model, optimizer, args)
        save_weights(model, args)
