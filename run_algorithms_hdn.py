import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import pickle
import json

from algorithms_hdn import jpmap_hdn, csgm_hdn, get_zdim_hdn, x_step_A, x_step_Denoising, x_step_MissingPixels
from utils import *
from hdn_utils import load_hdn_model, get_normalized_tensor, enc_dec_pass, dec_pass

parser = argparse.ArgumentParser()
parser.add_argument('--models-folder', type=str, default='./pretrained_models')
parser.add_argument('--model', type=str, default='HDN_natural')
parser.add_argument('--output-path', type=str, default='./experiments')
parser.add_argument('--dataset', type=str, default='natural')
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--gpu', action='store_true', default=True)
parser.add_argument('--exp-name', type=str, default='denoising_01')
parser.add_argument('--problem', type=str, default='denoising')  # 'compsensing', 'denoising', 'missingpixels'
parser.add_argument('--sigma', type=int, default=5)  # Noise std in [0,255]
parser.add_argument('--model_noise', type=int, default=5)  # Model noise std of HDN in [0,255]
parser.add_argument('--output-dim', type=int, default=100)  # Number of measurements in Compressed Sensing
parser.add_argument('--iters-jpmap', type=int, default=350)  # Maximum number of JPMAP iterations
parser.add_argument('--missing', type=float, default=0.8)  # Percentage of Missing Pixels
parser.add_argument('--n-samples', type=int, default=1)  # Number of restored images
parser.add_argument('--range-dec', action='store_true', default=False)  # Take ground truth from decoder range?
parser.add_argument('--uzawa', action='store_true',
                    default=False)  # Use Exponential Multiplier Method (Uzawa) for beta update?
parser.add_argument('--alpha', type=float, default=5)  # Parameter of Uzawa algorithm
parser.add_argument('--rho', type=float, default=100)  # Parameter of Uzawa algorithm
parser.add_argument('--save-iters', action='store_true', default=False)  # Save JPMAP iterations?

args = parser.parse_args()
args.cuda = args.gpu and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print('(running on device %s)' % device)
torch.manual_seed(1.)  # For experiment reproducibility

# Experiment and Model folders
exp_folder = os.path.join(args.output_path, args.dataset, args.exp_name)
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
model_path = os.path.join(args.models_folder, args.model)

# Load model
print('device :', device)
hdn = load_hdn_model(model_path, device=device)

# Load target images (from test dataset)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
test_loader = load_dataset(args.dataset, args.n_samples, kwargs, args.dataset_root)[1]
x_target = next(iter(test_loader))[0].to(device)

x_size = x_target.size()[1:]  # Image shape
n_pixels = x_target[0].nelement()  # Total number of pixels
n_channels = x_target.size(1)  # [N,C,H,W] format
sigma = args.sigma / 255.

# Construct forward operator (degradation model)
if args.problem == 'compsensing':
    output_dim = args.output_dim
    A = (1. / np.sqrt(output_dim)) * torch.randn(output_dim, n_pixels, device=device)  # .astype(np.float32)
    x_step = x_step_A

elif args.problem == 'denoising':
    output_dim = n_pixels
    A = torch.eye(n_pixels, device=device)  # , dtype=np.float32)
    x_step = x_step_Denoising

elif args.problem == 'missingpixels':
    output_dim = n_pixels
    missing_pixels = args.missing
    if n_channels == 1:  # 1 channel = Grayscale image
        mask = 1. * (torch.rand(n_pixels) > missing_pixels)
    elif n_channels == 3:  # 3 channels = RGB image
        mask = 1. * (torch.rand(n_pixels // 3) > missing_pixels)
        mask = torch.cat(3 * [mask], dim=0).view(-1)
    A = torch.diag(mask).to(device)  # .astype(np.float32)
    x_step = x_step_MissingPixels


_, x_reconst = enc_dec_pass(x_target, hdn)

if args.range_dec:
    x_target = x_reconst  # ground truth from decoder range

x_noisy = torch.zeros((args.n_samples, output_dim))
for i in range(args.n_samples):
    x_noisy[i, :] = torch.matmul(A, x_target[i, :, :, :].view(-1))

x_noisy += sigma * torch.randn(*x_noisy.shape)
x_noisy = x_noisy.to(device)

############################################################################################################################

x_results_csgm = torch.zeros(x_target.shape, device=device)
x_results_jpmap = torch.zeros(x_target.shape, device=device)

mse_results = {'csgm': [], 'jpmap': []}

if args.save_iters:
    outfile = os.path.join(exp_folder, 'A.npy')
    np.save(outfile, A.cpu())

for ind in range(args.n_samples):

    y = x_noisy[ind, :]
    xtarget = x_target[ind, :]  # Only to compute MSE to ground truth (not mandatory)

    # Verbose
    print()
    print('########################################################################################')
    print('Running experiment %s  ||  Restoring image %d of %d...' % (args.exp_name, ind + 1, args.n_samples))
    print('########################################################################################')
    print()


    with torch.no_grad():
        zdim, total_dim = get_zdim_hdn(xtarget[None,:], hdn)

    # Same initialization for both algorithms
    xinit = torch.matmul(A.T, y).view(1, *x_size)  # JPMAP initialization
    zinit = [torch.zeros(zdim[i]).to(device) for i in range(len(zdim))]  # CSGM initialization

    # CSGM (Bora et al.)
    args.csgm_lamb = sigma ** 2
    print('sigma =', sigma)
    print('csgm_lamb =', args.csgm_lamb)
    [xopt_csgm, zopt_csgm] = csgm_hdn(y, A, hdn, args.csgm_lamb, xtarget=xtarget, zinit=zinit, device=device)
    x_results_csgm[ind, :] = xopt_csgm.detach()
    mse_results['csgm'].append(compute_mse(xtarget, xopt_csgm))

    # JPMAP
    params = {'alpha': args.alpha, 'rho': args.rho}  # Exponential multiplier method (Uzawa algorithm)
    output_jpmap = jpmap_hdn(y, x_size, hdn, A, x_step, sigma, args.model_noise / 255, xinit=xinit, xtarget=xtarget, max_iters=args.iters_jpmap,
                         uzawa=args.uzawa, device=device, save_iters=args.save_iters, params=params)
    [xopt_jpmap, zopt_jpmap] = output_jpmap[0:2]

    x_results_jpmap[ind, :] = xopt_jpmap.detach()
    mse_results['jpmap'].append(compute_mse(xtarget, xopt_jpmap))

    if args.save_iters:
        outfile = os.path.join(exp_folder, 'xiters_jpmap_%2d.npy' % ind)
        np.save(outfile, output_jpmap[2])

        outfile = os.path.join(exp_folder, 'ziters_jpmap_%2d.pkl' % ind)
        with open(outfile, "wb") as f:
            pickle.dump(output_jpmap[3], f)

        outfile = os.path.join(exp_folder, 'beta_k_%2d.npy' % ind)
        np.save(outfile, output_jpmap[4])

        outfile = os.path.join(exp_folder, 'indices_%2d.npy' % ind)
        np.save(outfile, output_jpmap[5])

        outfile = os.path.join(exp_folder, 'y_%2d.npy' % ind)
        np.save(outfile, y.cpu())

        outfile = os.path.join(exp_folder, 'xtarget_%2d.npy' % ind)
        np.save(outfile, xtarget.cpu())

        outfile = os.path.join(exp_folder, 'xopt_csgm_%2d.npy' % ind)
        np.save(outfile, xopt_csgm.cpu().detach().numpy())

    ## Save experiment's parameters and results
    if args.problem != 'compsensing':
        output_image = torch.cat([x_target, x_noisy.view(args.n_samples, *x_size), x_results_csgm, x_results_jpmap])
    else:
        output_image = torch.cat([x_target, x_results_csgm, x_results_jpmap])
    save_image(output_image.cpu(), os.path.join(exp_folder, 'results_%s.png' % args.exp_name), nrow=args.n_samples)

    with open(os.path.join(exp_folder, 'experiment_args.json'), 'w') as outfile:
        json.dump(vars(args), outfile)

    with open(os.path.join(exp_folder, 'mse_results.json'), 'w') as outfile:
        json.dump(mse_results, outfile)
