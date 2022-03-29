import os

import torch
from lib.utils import crop_img_tensor


def load_hdn_model(path, device='cuda'):

    model = torch.load(os.path.join(path, 'model.net'),  map_location=device)
    model.mode_pred = True
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def get_normalized_tensor(test_images,model):
    '''
    Normalizes tensor with mean and std.
    Parameters
    ----------
    img: array
        Image.
    model: Hierarchical DivNoising model
    device: GPU device.
    '''

    test_images = 255*test_images  # because img in [0,1] for JPMAP
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images-data_mean)/data_std
    return test_images


def get_unnormalized_tensor(test_images,model):
    data_mean = model.data_mean
    data_std = model.data_std

    test_images = test_images * data_std + data_mean
    test_images /= 255  # because img in [0,1] for JPMAP
    return test_images


def enc_dec_pass(test_img, hdn):
    """

    Parameters
    ----------
    test_img: shape (w,h)
    hdn
    device

    Returns
    -------

    """
    width, height = test_img.shape[-2:]
    bs = test_img.shape[0]
    test_img = get_normalized_tensor(test_img.view((bs,1, width, height)), hdn)
    output = hdn(test_img)
    zE = output['z']
    mu_D = output['out_mean']
    mu_D = get_unnormalized_tensor(mu_D, hdn)
    return zE, mu_D


def dec_pass(z_tab, hdn, img_shape):

    mu_D, data = hdn.topdown_pass(forced_latent=z_tab, n_img_prior=1)
    mu_D = crop_img_tensor(mu_D, img_shape)
    mu_D = hdn.likelihood.distr_params(mu_D)['mean']
    mu_D = get_unnormalized_tensor(mu_D, hdn)

    return mu_D, data


