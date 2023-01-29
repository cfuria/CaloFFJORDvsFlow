""" This classifier should learn the difference between GEANT and CaloFlow (CaloGAN) events.
    If it is unable to tell the difference, the generated samples are realistic.

    Used for
    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

######################################   Imports   #################################################
import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from data import get_dataloader

torch.set_default_dtype(torch.float64)

#####################################   Parser setup   #############################################
parser = argparse.ArgumentParser()

# file structures
parser.add_argument('--save_dir', default='./classifier', help='Where to save the trained model')
parser.add_argument('--data_dir', default='/media/claudius/8491-9E93/ML_sources/CaloGAN/classifier',
                    help='Where to find the dataset')

# Calo specific
parser.add_argument('--particle_type', '-p',
                    help='Name of the dataset file w/o extension and "train_" or "test_" prefix')
parser.add_argument('--which_layer', default='all',
                    help='Which layers to compare. One of ["0", "1", "2", "all"]')

# NN parameters

# DNN or CNN
parser.add_argument('--mode', default='DNN',
                    help='must be in ["DNN", "CNN", "ROC"]')
parser.add_argument('--n_layer', type=int, default=2,
                    help='Number of hidden layers in the classifier.')
parser.add_argument('--n_hidden', type=int, default='512',
                    help='Hidden nodes per layer.')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.,
                    help='dropout probability')
parser.add_argument('--use_logit', action='store_true', help='If data is logit transformed')
parser.add_argument('--threshold', action='store_true', help='If threshold of 1e-2MeV is applied')
parser.add_argument('--normalize', action='store_true',
                    help='If voxels should be normalized per layer')

# training params
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=70,
                    help='How often to show loss statistics.')
parser.add_argument('--load', action='store_true', default=False,
                    help='Whether or not load model from --save_dir')

# CUDA parameters
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

#######################################   helper functions   #######################################
ALPHA = 1e-6
def logit(x):
    return torch.log(x / (1.0 - x))

def logit_trafo(x):
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def make_input(dataloader, args):
    """ takes dataloader and returns tensor of
        layer0, layer1, layer2, log10 energy
    """
    layer0 = dataloader['layer_0']
    layer1 = dataloader['layer_1']
    layer2 = dataloader['layer_2']
    energy = torch.log10(dataloader['energy']*10.).to(args.device)
    E0 = dataloader['layer_0_E']
    E1 = dataloader['layer_1_E']
    E2 = dataloader['layer_2_E']

    if args.threshold:
        layer0 = torch.where(layer0 < 1e-7, torch.zeros_like(layer0), layer0)
        layer1 = torch.where(layer1 < 1e-7, torch.zeros_like(layer1), layer1)
        layer2 = torch.where(layer2 < 1e-7, torch.zeros_like(layer2), layer2)

    if args.normalize:
        layer0 /= (E0.reshape(-1, 1, 1) +1e-16)
        layer1 /= (E1.reshape(-1, 1, 1) +1e-16)
        layer2 /= (E2.reshape(-1, 1, 1) +1e-16)

    E0 = (torch.log10(E0.unsqueeze(-1)+1e-8) + 2.).to(args.device)
    E1 = (torch.log10(E1.unsqueeze(-1)+1e-8) + 2.).to(args.device)
    E2 = (torch.log10(E2.unsqueeze(-1)+1e-8) + 2.).to(args.device)

    target = dataloader['label'].to(args.device)

    layer0 = layer0.view(layer0.shape[0], -1).to(args.device)
    layer1 = layer1.view(layer1.shape[0], -1).to(args.device)
    layer2 = layer2.view(layer2.shape[0], -1).to(args.device)

    if args.use_logit:
        layer0 = logit_trafo(layer0)/10.
        layer1 = logit_trafo(layer1)/10.
        layer2 = logit_trafo(layer2)/10.

    return torch.cat((layer0, layer1, layer2, energy, E0, E1, E2), 1), target

def load_classifier(constructed_model, parser_args, filepath=None):
    """ loads a saved model """
    if filepath is None:
        checkpoint = torch.load(os.path.join(parser_args.save_dir, parser_args.mode+'.pt'),
                                map_location=args.device)
    else:
        checkpoint = torch.load(filepath,
                                map_location=args.device)
    constructed_model.load_state_dict(checkpoint['model_state_dict'])
    constructed_model.to(parser_args.device)
    constructed_model.eval()
    return constructed_model

######################################## constructing the NN #######################################

class DNN(torch.nn.Module):
    """ NN for vanilla classifier """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.,
                 is_classifier=True):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.ReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.ReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        if is_classifier:
            all_layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class CNN(torch.nn.Module):
    """ CNN for improved classification """
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers_0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=(3, 7), padding=(2, 0)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=(4, 7), padding=(2, 0)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 14)),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.cnn_layers_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.cnn_layers_2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(256+256+256+4, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        split_elements = [*torch.split(x, [288, 144, 72, 4], dim=-1)]
        layer_0 = split_elements[0].view(x.size(0), 3, 96)
        layer_1 = split_elements[1].view(x.size(0), 12, 12)
        layer_2 = split_elements[2].view(x.size(0), 12, 6)
        layer_0 = self.cnn_layers_0(layer_0.unsqueeze(1)).view(x.size(0), -1)
        layer_1 = self.cnn_layers_1(layer_1.unsqueeze(1)).view(x.size(0), -1)
        layer_2 = self.cnn_layers_2(layer_2.unsqueeze(1)).view(x.size(0), -1)
        all_together = torch.cat((layer_0, layer_1, layer_2, split_elements[3]), 1)
        ret = self.linear_layers(all_together)
        return ret

##################################### train and evaluation functions ###############################
def train_and_evaluate(model, data_train, data_test, optimizer, args):
    """ train the model and evaluate along the way"""
    best_eval_acc = float('-inf')
    for i in range(args.n_epochs):
        train(model, data_train, optimizer, i, args)
        if i >= 19 or i % 5 == 0:
            with torch.no_grad():
                eval_acc, _ = evaluate(model, data_test, i, args)
                #args.test_loss.append(-eval_loss.to('cpu').numpy())
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save({'model_state_dict':model.state_dict()},
                           os.path.join(args.save_dir, args.mode+'.pt'))

def train(model, train_data, optimizer, epoch, args):
    """ train one step """
    model.train()
    #res_true = []
    #res_pred = []
    for i, data_batch in enumerate(train_data):
        input_vector, target_vector = make_input(data_batch, args)

        output_vector = model(input_vector)
        criterion = torch.nn.BCELoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (len(train_data)//2) == 0:
            print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, args.n_epochs, i, len(train_data), loss.item()))
        # PREDICTIONS
        #pred = np.round(output_vector.detach().cpu())
        #target = np.round(target_vector.detach().cpu())
        #res_pred.extend(pred.tolist())
        #res_true.extend(target.tolist())
        pred = torch.round(output_vector.detach())
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    print("Accuracy on training set is",
          accuracy_score(res_true.cpu(), res_pred.cpu()))
          #accuracy_score(res_true.detach().cpu(), res_pred.detach().cpu()))

def evaluate(model, test_data, i, args, return_ROC=False):
    """ evaluate on test set """
    model.eval()
    #result_true = []
    #result_pred = []
    eval_loss = []
    for j, data_batch in enumerate(test_data):
        input_vector, target_vector = make_input(data_batch, args)
        output_vector = model(input_vector)
        #criterion = torch.nn.BCELoss()
        #eval_loss.append(criterion(output_vector, target_vector.unsqueeze(1)).unsqueeze(-1))

        #pred = np.round(output_vector.cpu())

        #pred = output_vector.cpu()
        #target = target_vector.float()
        #result_true.extend(target.tolist())
        #result_pred.extend(pred.reshape(-1).tolist())
        pred = output_vector.reshape(-1)
        target = target_vector.float()
        if j == 0:
            result_true = target
            result_pred = pred

            log_A = torch.log(pred)[target == 1]
            log_B = torch.log(1.-pred)[target == 0]
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)

            log_A = torch.cat((log_A, torch.log(pred)[target == 1]), 0)
            log_B = torch.cat((log_B, torch.log(1.-pred)[target == 0]), 0)
    #eval_loss_tot = torch.cat((eval_loss), dim=0).to(args.device)

    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    eval_acc = accuracy_score(result_true, np.round(result_pred))
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, result_pred)
    print("AUC on test set is", eval_auc)
    BCE = torch.mean(log_A) + torch.mean(log_B)
    JSD = 0.5* BCE + np.log(2.)
    print("BCE loss of test set is {}, JSD of the two dists is {}".format(-BCE, JSD/np.log(2.)))
    if not return_ROC:
        return eval_acc, JSD #, eval_loss_tot.mean(0)
    else:
        return roc_curve(result_true, result_pred)

####################################################################################################
#######################################   running the code   #######################################
####################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    # check if parsed arguments are valid
    assert args.which_layer in ['0', '1', '2', 'all'], (
        "--which_layer can only be '0', '1', '2', or 'all'")
    assert args.mode in ['DNN', 'CNN', 'ROC'], (
        '--mode must be in ["DNN", "CNN", "ROC"]')

    # set up device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    input_dim = {'0': 288, '1': 144, '2': 72, 'all': 504}[args.which_layer]
    input_dim += 4
    DNN_kwargs = {'num_layer':args.n_layer,
                  'num_hidden':args.n_hidden,
                  'input_dim':input_dim,
                  'dropout_probability':args.dropout_probability}
    if args.mode == 'DNN':
        model = DNN(**DNN_kwargs)
    elif args.mode == 'CNN':
        model = CNN()
    else:
        model_DNN = DNN(**DNN_kwargs)
        model_CNN = CNN()
        model_DNN.to(args.device)
        model_CNN.to(args.device)
    if args.mode != 'ROC':
        model.to(args.device)
        print(model)
        total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("{} has {} parameters".format(args.mode, int(total_parameters)))

    if args.mode in ['DNN', 'CNN']:
        data_train, data_test = get_dataloader(args.particle_type,
                                               args.data_dir,
                                               full=False,
                                               apply_logit=False,
                                               device=args.device,
                                               batch_size=args.batch_size,
                                               with_noise=False,
                                               normed=False,
                                               normed_layer=False,
                                               return_label=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        args.test_loss = []

        if os.path.exists(os.path.join(args.save_dir, args.mode+'.pt')) and args.load:
            model = load_classifier(model, args)
        else:
            train_and_evaluate(model, data_train, data_test, optimizer, args)

        model = load_classifier(model, args)
        with torch.no_grad():
            eval_logprob = evaluate(model, data_test, 0, args)
    else:
        paths_to_files = []
        labels = []
        for class_type in ['DNN', 'CNN']:
            for p_type in ['eplus', 'gamma', 'piplus']:
                paths_to_files.append(os.path.join('./classifier',
                                                   class_type+\
                                                   ('_no' if not args.normalize else '')+\
                                                   '_normalized',
                                                   'thres' if args.threshold else 'no_thres',
                                                   'logit' if args.use_logit else 'no_logit',
                                                   p_type+\
                                                   '_CaloFlow',
                                                   class_type+'.pt'))
                labels.append(paths_to_files[-1])
                paths_to_files.append(os.path.join('./classifier',
                                                   class_type+\
                                                   ('_no' if not args.normalize else '')+\
                                                   '_normalized',
                                                   'thres' if args.threshold else 'no_thres',
                                                   'logit' if args.use_logit else 'no_logit',
                                                   p_type+\
                                                   '_CaloGAN',
                                                   class_type+'.pt'))
                labels.append(paths_to_files[-1])

        ROC_curves = []
        for i, path in enumerate(paths_to_files):
            if 'eplus' in path:
                p_type = 'eplus'
            elif 'gamma' in path:
                p_type = 'gamma'
            elif 'piplus' in path:
                p_type = 'piplus'
            if 'CaloFlow' in path:
                particle_file = 'merged_'+p_type +'_CaloFlow_'+p_type+'_2'
            else:
                particle_file = 'merged_'+p_type +'_CaloGAN_'+p_type
                if p_type == 'piplus':
                    particle_file += '_2'
            print("using file ", particle_file)
            _, data_test = get_dataloader(particle_file,
                                          args.data_dir,
                                          full=False,
                                          apply_logit=False,
                                          device=args.device,
                                          batch_size=args.batch_size,
                                          with_noise=False,
                                          normed=False,
                                          normed_layer=False,
                                          return_label=True)

            if 'DNN' in path:
                model = model_DNN
            else:
                model = model_CNN
            model = load_classifier(model, args, path)
            print('loaded from ', path)
            with torch.no_grad():
                ROC_curves.append(evaluate(model, data_test, 0, args, return_ROC=True))
            print("Done with {} / {}".format(i+1, len(paths_to_files)))

        plt.figure(figsize=(14, 10))
        colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
        matplotlib.rcParams.update({'font.size': 24})

        for i, entry in enumerate(ROC_curves):
            if 'eplus' in labels[i]:
                color = colors[0]
                plot_label = r'$e^+$ '
            elif 'gamma' in labels[i]:
                color = colors[1]
                plot_label = r'$\gamma$ '
            elif 'piplus' in labels[i]:
                color = colors[2]
                plot_label = r'$\pi^+$ '
            if 'GAN' in labels[i]:
                alpha = 0.75
                lw = 3.
                plot_label += 'CaloGAN '
            elif 'Flow' in labels[i]:
                alpha = 1.
                lw = 2.
                plot_label += 'CaloFlow '
            if 'DNN' in labels[i]:
                style = 'solid'
                plot_label += 'DNN'
            elif 'CNN' in labels[i]:
                style = 'dashed'
                plot_label += 'CNN'

            plt.plot(entry[0], entry[1], label=plot_label, color=color, lw=lw,
                     ls=style, alpha=alpha)

        plt.plot([0., 1.], [0., 1.], ls='dotted', color='k')
        plt.xlim([-0.01, 1.01])
        plt.xlabel('False positive rate')

        plt.ylim([-0.01, 1.01])
        plt.ylabel('True positive rate')

        legend_elements = [Line2D([0], [0], color='gray', lw=2, ls='solid', label='DNN'),
                           Line2D([0], [0], color='gray', lw=2, ls='dashed', label='CNN'),
                           Patch(facecolor=colors[0], label=r'$e^+$'),
                           Patch(facecolor=colors[1], label=r'$\gamma$'),
                           Patch(facecolor=colors[2], label=r'$\pi^+$')]

        #plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 0.9), loc='upper left')
        plt.legend(bbox_to_anchor=(1.01, 0.9), loc='upper left')

        title = 'ROC curve, '
        title +=  'normalized layer, ' if args.normalize else 'no normalized layer, '
        title +=  'w/ threshold, ' if args.threshold else 'w/o threshold, '
        title +=  'w/ logit' if args.use_logit else 'w/o logit'
        plt.title(title, fontsize=16)

        filename = 'ROC_'
        filename += 'normalized_' if args.normalize else 'no_normalized_'
        filename += 'thres_' if args.threshold else 'no_thres_'
        filename += 'logit.png' if args.use_logit else 'no_logit.png'
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, filename), dpi=300)
        plt.close()
