import os
import numpy as np
import torch
from torch import optim
from hdn_utils import enc_dec_pass, dec_pass


def z_step_hdn(xk, hdn, beta, zdim, zinit, max_iters=1000, tol=1e-4, lr_adam=0.1, verbose=False):
    if verbose:
        print('Running z_step...')
        print('lr_adam = ', lr_adam)

    zk = [zinit[i].detach().clone().requires_grad_(True) for i in range(len(zdim))]

    # Optimizer
    # lr_adam = 1e-2
    lr_adam = 0.1
    optimizer = optim.Adam(zk, lr=lr_adam)

    convergence = False
    k = 0

    while not convergence and k < max_iters:
        k += 1
        optimizer.zero_grad()

        # lossF = (1/2)*||z||^2 + (beta/2)*||xk - Dec(z)||^2
        # (beta/2)*||xk - Dec(z)||^2
        g_z, data = dec_pass(zk, hdn, xk.shape[-2:])
        Acople = 0.5 * torch.sum((xk - g_z[0]).pow(2))

        # prior
        p_params = data['p_params']
        Prior = 0
        for i in range(len(p_params)):
            p_mu, p_lv = p_params[i].chunk(2, dim=1)
            p_sigma = (p_lv / 2).exp()
            Prior += (0.5 / beta) * torch.sum(((zk[i] - p_mu) / p_sigma).pow(2))

        loss = Acople + Prior
        loss.backward()
        grad_zk = torch.tensor([torch.norm(zk[i].grad / torch.numel(zk[i])) for i in range(len(zdim))])
        grad_index = torch.argmax(grad_zk)
        grad = grad_zk[grad_index]  # = max_(||grad(z_k)||)

        if k % 100 == 1 and verbose:
            print(f"iter : {k} -- grad : {grad.item()} -- tol {tol} -- acople {Acople.item()} -- prior {Prior.item()}")

        if grad.item() < tol:
            convergence = True
        else:
            optimizer.step()

    if verbose:
        print(
            'Adam terminated in %d iterations (out of %d) (z-step JPMAP)  |  norm(grad)/zdim at last iteration (z-step JPMAP): %.4f' % (
                k, max_iters, torch.norm(grad) / zdim))

    return [zk[i].data for i in range(len(zdim))]


# Generic x-step:
def x_step_A(y, x_shape, sigma, A, mu_D, beta, verbose=False):
    # OUTPUT:
    #        x step:  argmin_x (1/2*sigma^2)*||Ax-y||^2 + D(x, z^k)

    if verbose:
        print('Running x_step_A...')

    mu_D = mu_D.view(-1)
    Amatrix = torch.matmul(A.t(), A) / beta + (sigma ** 2) * torch.eye(mu_D.nelement(), out=torch.empty_like(A))
    btensor = torch.matmul(A.t(), y) / beta + (sigma ** 2) * mu_D

    x_new = torch.solve(btensor.view(-1, 1), Amatrix)[0]

    return x_new.view(x_shape)  # .astype(xtilde.dtype)


# Efficient x-step: Denoising
def x_step_Denoising(y, x_shape, sigma, A, mu_D, beta, verbose=False):
    if verbose:
        print('Running x_step_Denoising...')
    mu_D = mu_D.view(-1)
    vect = y / beta + (sigma ** 2) * mu_D
    x_new = vect / (1 / beta + sigma ** 2)

    return x_new.view(x_shape)  # .astype(xtilde.dtype)


# Efficient x-step: Missing Pixels
def x_step_MissingPixels(y, x_shape, sigma, A, mu_D, beta, verbose=False):
    if verbose:
        print('Running x_step_MissingPixels...')
    mask = torch.diag(A).view(-1)
    mu_D = mu_D.view(-1)
    vect = mask * y / beta + (sigma ** 2) * mu_D
    den = mask / beta + sigma ** 2
    x_new = vect / den

    return x_new.view(x_shape)  # .astype(xtilde.dtype)


def get_zdim_hdn(xinit, hdn):  # to get the dimensions of the latent variable z=(z_1,...,z_n)

    with torch.no_grad():
        output = hdn(xinit)

    z_sampled = output['z']
    zdim = []
    total_zdim = 0
    for z in z_sampled:
        zdim.append(z.shape)
        total_zdim += torch.numel(z)

    return zdim, total_zdim



def jpmap_hdn(y, x_shape, hdn, A, x_step, sigma, sigma_model, xtarget=None, max_iters=500, max_iters_inner=500, verbose=True, xinit=None,
          uzawa=False, device='cuda', save_iters=False, params=None):
    # Setup
    n = x_shape.numel()  # Number of pixels
    hdn = hdn.to(device)
    zdim, total_zdim = get_zdim_hdn(xinit, hdn)
    hdn.mode_pred = True

    with torch.no_grad():
        gamma = sigma_model
    print('gamma =', gamma)

    # A full matrix:
    if xinit is None:
        xk = torch.matmul(A.T, y).view(x_shape)
    else:
        xk = xinit.view(x_shape)

    zinit = [torch.zeros(zdim[i], device=device) for i in range(len(zdim))]
    zk = [zinit[i].requires_grad_(True) for i in range(len(zdim))]

    # Adam parameters (for z step)
    max_iters_z = 1000

    ## Exponential multiplier method (Uzawa)
    if uzawa:
        beta = 0.1 / gamma ** 2  # beta inicial
        rho = params['rho'] / n
        alpha = (params['alpha'] / 255) ** 2 * n
    else:
        beta = 1 / gamma ** 2  # Diagonal variance of decoder

    beta = 0.001 * beta  # need to do that otherwise it does not converge

    terminate = 0  # Main loop flag
    k = 0  # Iteration counter (outer loop)
    k_inner = 0  # Iteration counter (inner loop)

    x = xk
    z = zk
    J1_prev = np.Inf

    if save_iters:
        xiters_jpmap = np.array(xk.cpu().detach().clone())  # MNIST
        ziters_jpmap = [zk[i].cpu().detach().clone() for i in range(len(zdim))]
        beta_k = [beta]
        indices = []
        ind_k = []

    # Main loop
    while not terminate:
        k_inner += 1
        tol = 1 / 255  # 1/255 corresponds to a MSE of 1 gray level

        zE, mu_D_E = enc_dec_pass(xk[None,:], hdn)
        # mu_D_E = torch.clip(mu_D_E, min=0, max=1)
        xE = x_step(y, x_shape, sigma, A, mu_D_E, beta)
        J1_E = J1_hdn(xE, zE, A, y, sigma, beta, hdn, x_shape[1:])

        if J1_E < J1_prev:
            zk = zE
            xk = xE
            mu_D = mu_D_E
            J1_prev = J1_E
            if save_iters:
                ind_k.append(0)

        else:
            zE_GD = z_step_hdn(xk, hdn, beta, zdim, zinit=zE, max_iters=max_iters_z)
            mu_D_E_GD, _ = dec_pass(zE_GD, hdn, x_shape[1:])
            xE_GD = x_step(y, x_shape, sigma, A, mu_D_E_GD, beta)
            J1_E_GD = J1_hdn(xE_GD, zE_GD, A, y, sigma, beta, hdn, x_shape[1:])

            if J1_E_GD < J1_prev:
                zk = zE_GD
                xk = xE_GD
                mu_D = mu_D_E_GD
                J1_prev = J1_E_GD
                if save_iters:
                    ind_k.append(1)

            else:
                zk_GD = z_step_hdn(xk, hdn, beta, zdim, zinit=zk, max_iters=max_iters_z)
                mu_D_k_GD, _ = dec_pass(zk_GD, hdn, x_shape[1:])
                xk_GD = x_step(y, x_shape, sigma, A, mu_D_k_GD, beta)
                J1_k_GD = J1_hdn(xk_GD, zk_GD, A, y, sigma, beta, hdn, x_shape[1:])

                zk = zk_GD
                xk = xk_GD
                mu_D = mu_D_k_GD
                J1_prev = J1_k_GD
                if save_iters:
                    ind_k.append(2)

        ### Convergence criterion
        delta_x = torch.norm(x - xk) / np.sqrt(n) if k_inner != 0 else np.Inf
        delta_z = 0

        for i in range(len(zdim)):
            delta_z += torch.sum(torch.square(z[i] - zk[i]))

        delta_z = torch.sqrt(delta_z) / np.sqrt(total_zdim) if k_inner != 0 else np.Inf
        delta = delta_x + delta_z

        if verbose:
            print('ITER %d -->  ' % k,
                  'beta = %.4f  |  Delta_x: %.5f  |  Delta_z: %.5f  |  MSE to ground-truth: %.5f' % (
                  beta, delta_x, delta_z, compute_mse(xtarget, xk)))

        # Update iters
        x = xk
        z = zk

        if save_iters:
            print('Storing limit points x^k_infty Y z^k_infty...')
            xiters_jpmap = np.vstack([xiters_jpmap, xk.cpu().detach().clone()])
            ziters_jpmap = [np.vstack([ziters_jpmap[i], zk[i].cpu().detach().clone()]) for i in range(len(zdim))]
            beta_k.append(beta)
            indices.append(ind_k)
            ind_k = []

        if (k_inner >= max_iters_inner) or (delta < tol):

            # Update beta
            if uzawa:
                with torch.no_grad():
                    exp_beta = torch.exp(rho * (torch.norm(x - mu_D).pow(2) - alpha))
                    beta = (beta * exp_beta).item()

            k += 1  # Increment iter counter of outer loop
            k_inner = 0  # Reset iter counter of inner loop

            if (k >= max_iters) or (delta < tol / 100):
                terminate = 1

    if verbose:
        print('Restoration terminated in %d iterations (out of %d)' % (k, max_iters))

    if save_iters:
        return x, z, xiters_jpmap, ziters_jpmap, beta_k, indices
    else:
        return x, z


def csgm_hdn(y, A, hdn, lamb=1, max_iters=2000, tol=1e-5, xtarget=None, zinit=None, device='cuda', verbose=True,
             pix_value=False):
    # Bora, A., Jalal, A., Price, E., & Dimakis, A. G. (2017, August).
    # Compressed sensing using generative models. In Proceedings of
    # the 34th International Conference on Machine Learning-Volume 70
    # (pp. 537-546). JMLR. org.  https://arxiv.org/abs/1703.03208
    #
    # Computes G(argmin_z ||A*G(z) - y||^2 + lamb*||z||^2)
    if verbose:
        print('Running CSGM algorithm...')

    zdim, total_dim = get_zdim_hdn(xtarget[None,:], hdn)
    hdn = hdn.to(device)
    hdn.mode_pred = True
    x_shape =  xtarget.shape[-2:]

    ### CSGM's loss function:
    def csgm_loss(z, y, A, hdn, lamb, iter=None):

        ## Datafit = ||A*G(z) - y||^2
        dec_z, data = dec_pass(z, hdn, x_shape)
        AdotGz = torch.matmul(A, dec_z.view(-1))
        Datafit = torch.sum((AdotGz - y).pow(2))

        ## Reg = ||z||^2
        p_params = data['p_params']
        Reg = 0
        for i in range(len(p_params)):
            p_mu, p_lv = p_params[i].chunk(2, dim=1)
            p_sigma = (p_lv / 2).exp()
            Reg += torch.sum(torch.square((z[i] - p_mu) / p_sigma))

        ## lossF = ||A*Dec(z) - xtilde||^2 + lambda*||z||^2
        if iter%20 == 1:
            print(f"datafit {Datafit} -- reg {lamb * Reg}")
        return (Datafit + lamb * Reg)

    # Initialization
    if zinit is None:
        zinit = torch.zeros(zdim, device=device)

    activated_grad = len(zdim) * [True]
    zk = [zinit[i].requires_grad_(activated_grad[i]) for i in range(len(zdim)) if activated_grad[i]]
    convergence = False
    k = 0

    # Optimizer
    lr_adam = 0.01
    optimizer = optim.Adam(zk, lr=lr_adam)

    while not convergence and k < max_iters:

        k += 1
        optimizer.zero_grad()

        loss = csgm_loss(zk, y, A, hdn, lamb, k)
        loss.backward()
        grad_zk = torch.tensor([torch.norm(zk[i].grad / torch.numel(zk[i])) for i in range(len(zdim)) if activated_grad[i]])
        grad_index = torch.argmax(grad_zk)
        grad = grad_zk[grad_index]

        if grad < tol:
            convergence = True
        else:
            optimizer.step()
        if k%100 == 1:
            print(f'iter {k} -- grad {grad} -- tol {tol}')

    if verbose:
        print('Adam terminated in %d iterations (out of %d)' % (k, max_iters))
        print('norm(grad)/zdim at last iteration: %.4f' % (grad))

    xk = dec_pass(zk, hdn, x_shape)[0][0]

    if xtarget is not None and verbose:
        print(' --> MSE to ground-truth: %.5f' % compute_mse(xtarget, xk))

    return xk, zk


# Auxiliary functions
def compute_mse(x1, x2):
    N = x1.nelement()  # Number of pixels

    return float(torch.sum((x1 - x2).pow(2)) / N)

def J1_hdn(xE, zE, A, y, sigma, beta, hdn, x_shape):
    with torch.no_grad():
        # Datafit
        DataFit = (0.5 / sigma ** 2) * torch.sum((torch.matmul(A, xE.view(-1)) - y).pow(2))
        print("datafiit:", DataFit)

        # (beta/2)*||xk - Dec(z)||^2
        g_z, data = dec_pass(zE, hdn, x_shape)
        Acople = 0.5 * beta * torch.sum((xE - g_z[0]).pow(2))
        print('acope', Acople)

        # prior
        p_params = data['p_params']
        Prior = 0
        for i in range(len(p_params)):
            p_mu, p_lv = p_params[i].chunk(2, dim=1)
            p_sigma = (p_lv / 2).exp()
            Prior += 0.5 * torch.sum(((zE[i] - p_mu)/p_sigma).pow(2))
        print('prior', Prior)
    return DataFit + Acople + Prior


