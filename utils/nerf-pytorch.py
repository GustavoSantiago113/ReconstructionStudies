import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os, sys
import numpy as np
import imageio
import json
import random
import time
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import cv2
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
DEBUG = False

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(666)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            # pdb.set_trace();
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], 
                     -(j-K[1][2])/K[1][1], 
                     -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    

    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,  # noqa: E731
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    # capture a dictionary of hyperparameters with config
    # please uncomment the code starts with `wandb` to add your wandb project tracking
    # wandb.config = {"learning_rate": args.lrate, "iterations": args.training_iterations, 
    #                 "batch_size": 1024}

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : 0.0,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    # print("\n\n=----------------------------")
    # print("viewdirs: ",viewdirs)
    # print("pts: ", pts)
#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def modify_to_x_rotation(tensor):
    num_matrices = tensor.shape[0]  # Number of 3x5 matrices (120 in this case)
    modified_tensor = tensor.clone()

    for i in range(num_matrices):
        # Calculate the angle of rotation (linearly spaced from 0 to 2*pi)
        theta = torch.tensor(i * (2 * torch.pi / num_matrices))  # Convert to Tensor

        # Rotation matrix for y-axis
        R_y = torch.tensor([
            [torch.cos(theta), 0, torch.sin(theta)],
            [0, 1, 0],
            [-torch.sin(theta), 0, torch.cos(theta)]
        ])

        # Update the first 3x3 rotation matrix
        modified_tensor[i, :3, :3] = R_y

    return modified_tensor

# ---------------------------------------------------------------------------
# COLMAP conversion helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP quaternion (qw, qx, qy, qz) to a 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=float,
    )


def colmap_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP image pose to a 4×4 camera-to-world matrix.

    COLMAP stores the **world-to-camera** rigid transform:
        p_cam = R_cw @ p_world + t

    The inverse (camera-to-world) is:
        c2w[:3, :3] = R_cw.T
        c2w[:3,  3] = −R_cw.T @ t   (= camera centre in world space)
    """
    R_cw = qvec_to_rotmat(qvec)
    t = np.asarray(tvec, dtype=float)
    c2w = np.eye(4, dtype=float)
    c2w[:3, :3] = R_cw.T
    c2w[:3, 3] = -(R_cw.T @ t)
    return c2w


def colmap_intrinsics(cam: Dict) -> np.ndarray:
    """Build a 3×3 K matrix from a COLMAP camera info dictionary.

    Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, and RADIAL models.
    Distortion parameters are ignored (COLMAP's undistorted images should
    be used for training).  All other camera models fall back to a rough
    FOV≈60° estimate.
    """
    model = cam["model"].upper()
    params = cam["params"]
    W, H = cam["width"], cam["height"]

    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]; cx, cy = params[1], params[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in {"SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]; cx, cy = params[1], params[2]
    else:                    # rough FOV ≈ 60° fallback
        fx = fy = max(W, H) * 0.866
        cx, cy = W / 2.0, H / 2.0

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


# ---------------------------------------------------------------------------
# Standalone ray-generation helper (used by ColmapNeRFDataset and externally)
# ---------------------------------------------------------------------------

def get_rays(
    H: int,
    W: int,
    K: "torch.Tensor",   # (3, 3) intrinsic matrix
    c2w: "torch.Tensor", # (4, 4) camera-to-world matrix
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Return (rays_o, rays_d) tensors of shape (H, W, 3) for a pinhole camera."""
    device = K.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dirs = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1
    )  # (H, W, 3)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)                       # (H, W, 3)
    return rays_o, rays_d


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ColmapNeRFDataset:
    """Load a COLMAP binary reconstruction and matching RGB images for NeRF.

    Parameters
    ----------
    colmap_dir  : Sub-model folder containing ``cameras.bin``, ``images.bin``,
                  and ``points3D.bin``  (e.g. ``results/colmap_miniature/0``).
    image_dir   : Root directory where source images live.
    scale_factor: Resize images by this factor (<1 to downscale).  Smaller
                  images reduce memory and speed up training significantly.
    white_bg    : Composite over a white background during rendering.
    device      : ``"cpu"`` or ``"cuda"``.

    Attributes
    ----------
    images      : (N, H, W, 3) float32 tensor in [0, 1].
    c2w         : (N, 4, 4) float32 camera-to-world matrices (scene-normalised).
    K           : (N, 3, 3) float32 intrinsic matrices (adjusted for scale).
    near, far   : float – near/far plane distances in normalised scene units.
    scene_center: (3,) ndarray – world-space origin used for normalisation.
    scene_scale : float – world-space radius used for normalisation.
    rays_o      : (N·H·W, 3) float32 – pre-computed ray origins.
    rays_d      : (N·H·W, 3) float32 – pre-computed ray directions.
    targets     : (N·H·W, 3) float32 – ground-truth pixel colours.
    """

    def __init__(
        self,
        colmap_dir: str,
        image_dir: str,
        scale_factor: float = 1.0,
        white_bg: bool = True,
        device: str = "cpu",
    ) -> None:
        # Locate the colmap sibling module
        _here = Path(__file__).parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        from colmap import load_colmap_bin_model  # type: ignore

        cams_dict, images_dict, points_dict = load_colmap_bin_model(colmap_dir)
        image_dir_p = Path(image_dir)

        sorted_ids = sorted(images_dict.keys(), key=lambda k: images_dict[k]["name"])
        c2w_list, K_list, img_list = [], [], []

        for img_id in sorted_ids:
            info = images_dict[img_id]
            img_path = self._locate_image(image_dir_p, info["name"])
            if img_path is None:
                print(f"[ColmapNeRFDataset] Warning: '{info['name']}' not found – skipping.")
                continue
            img = Image.open(img_path).convert("RGB")
            if scale_factor != 1.0:
                nw = max(1, int(img.width  * scale_factor))
                nh = max(1, int(img.height * scale_factor))
                img = img.resize((nw, nh), Image.LANCZOS)
            img_list.append(np.array(img, dtype=np.float32) / 255.0)

            cam = cams_dict[info["camera_id"]]
            K = colmap_intrinsics(cam)
            if scale_factor != 1.0:
                K = K.copy()
                K[0] *= scale_factor   # scale fx, cx
                K[1] *= scale_factor   # scale fy, cy
            K_list.append(K)
            c2w_list.append(colmap_to_c2w(info["qvec"], info["tvec"]))

        if not img_list:
            raise RuntimeError(
                "No images loaded.  Verify that colmap_dir and image_dir are correct."
            )

        c2w_arr = np.stack(c2w_list).astype(np.float32)   # (N, 4, 4)

        # ── Scene normalisation ──────────────────────────────────────────────
        cam_centers = c2w_arr[:, :3, 3]                   # (N, 3)
        self.scene_center = cam_centers.mean(axis=0)
        radii = np.linalg.norm(cam_centers - self.scene_center, axis=-1)
        self.scene_scale = float(radii.max()) or 1.0
        c2w_arr[:, :3, 3] = (c2w_arr[:, :3, 3] - self.scene_center) / self.scene_scale

        # ── Near/far from the sparse point cloud ────────────────────────────
        if points_dict:
            rng = np.random.default_rng(0)
            pts = np.stack([v["xyz"] for v in points_dict.values()]).astype(np.float32)
            pts = (pts - self.scene_center) / self.scene_scale
            if len(pts) > 50_000:                    # subsample for speed
                pts = pts[rng.choice(len(pts), 50_000, replace=False)]
            # Per-camera distances to all (subsampled) points.  (N, M)
            dists = np.linalg.norm(pts[None] - c2w_arr[:, :3, 3][:, None], axis=-1)
            self.near = float(max(np.percentile(dists, 0.5),  0.01))
            self.far  = float(np.percentile(dists, 99.5))
        else:
            self.near, self.far = 0.1, 6.0

        # ── Convert to tensors ───────────────────────────────────────────────
        self.c2w    = torch.from_numpy(c2w_arr).to(device)
        self.K      = torch.from_numpy(np.stack(K_list).astype(np.float32)).to(device)
        imgs_np     = np.stack(img_list)               # (N, H, W, 3)
        self.images = torch.from_numpy(imgs_np).to(device)
        self.white_bg = white_bg
        self.device   = device
        self.N, self.H, self.W = imgs_np.shape[:3]

        print(
            f"[ColmapNeRFDataset] {self.N} images ({self.H}×{self.W})  "
            f"near={self.near:.3f}  far={self.far:.3f}  "
            f"scene_scale={self.scene_scale:.4f}"
        )
        self._build_rays()

    # -------------------------------------------------------------------------
    @staticmethod
    def _locate_image(image_dir: Path, name: str) -> Optional[Path]:
        """Find an image file using several fallback strategies."""
        for candidate in [image_dir / name, image_dir / Path(name).name]:
            if candidate.exists():
                return candidate
        # Recursive search as last resort
        hits = list(image_dir.rglob(Path(name).name))
        return hits[0] if hits else None

    def _build_rays(self) -> None:
        """Pre-compute all ray origins, directions, and target colours."""
        all_o, all_d, all_t = [], [], []
        for i in range(self.N):
            ro, rd = get_rays(self.H, self.W, self.K[i], self.c2w[i])
            all_o.append(ro.reshape(-1, 3))
            all_d.append(rd.reshape(-1, 3))
            all_t.append(self.images[i].reshape(-1, 3))
        self.rays_o  = torch.cat(all_o, dim=0)   # (N·H·W, 3)
        self.rays_d  = torch.cat(all_d, dim=0)
        self.targets = torch.cat(all_t, dim=0)

    def sample_rays(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample *batch_size* rays from all training images.

        Returns
        -------
        rays_o  : (B, 3)
        rays_d  : (B, 3)
        targets : (B, 3) ground-truth RGB
        """
        idx = torch.randint(0, self.rays_o.shape[0], (batch_size,), device=self.device)
        return self.rays_o[idx], self.rays_d[idx], self.targets[idx]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        """Return ``(image, c2w, K)`` for image *idx*."""
        return self.images[idx], self.c2w[idx], self.K[idx]


def train():

    parser = config_parser()
    args = parser.parse_args()

    # ── Load data from COLMAP reconstruction ────────────────────────────
    scale_factor = getattr(args, 'scale_factor', 1.0)
    dataset = ColmapNeRFDataset(
        colmap_dir=args.colmap_dir,
        image_dir=args.datadir,
        scale_factor=scale_factor,
        white_bg=getattr(args, 'white_bkgd', False),
        device='cpu',
    )

    H, W = dataset.H, dataset.W
    K = dataset.K[0].numpy()           # representative (3, 3) intrinsics for rendering
    focal = float(K[0, 0])
    hwf = [H, W, focal]
    near = dataset.near
    far  = dataset.far

    # ── Train / val / test split (every 8th image held out) ─────────────
    all_indices = list(range(dataset.N))
    i_test  = all_indices[::8]
    i_val   = i_test
    i_train = [idx for idx in all_indices if idx not in set(i_test)]

    images_np = dataset.images.numpy()   # (N, H, W, 3)  float32 [0, 1]
    poses_np  = dataset.c2w.numpy()      # (N, 4, 4)     float32

    # Render poses default to the held-out test cameras
    render_poses = poses_np[i_test]
    if args.render_test:
        render_poses = poses_np[i_test]

    # Cast intrinsics to right types
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Configure logging
    log_dir = os.path.join(basedir, expname, f"logs_{args.datadir.split('/')[-1]}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Logs to both file and console
        ]
    )

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move render poses to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            gt_imgs = images_np[i_test] if args.render_test else None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test,
                                      gt_imgs=gt_imgs, savedir=testsavedir,
                                      render_factor=args.render_factor)
            print('Done rendering', testsavedir)

            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps), fps=30, quality=8)

            # Apply heatmap colormap
            colored_disps = [plt.cm.inferno(d)[:, :, :3] for d in disps]  # Normalize and apply colormap
            colored_disps_8bit = [np.uint8(c * 255) for c in colored_disps]
            imageio.mimwrite(os.path.join(testsavedir, 'disp_heatmap.mp4'), colored_disps_8bit, fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    if use_batching:
        # Use dataset's pre-computed rays from training images only
        print('Building ray batch from training images...')
        train_rays_o  = dataset.rays_o.reshape(dataset.N, -1, 3)[i_train].reshape(-1, 3).to(device)
        train_rays_d  = dataset.rays_d.reshape(dataset.N, -1, 3)[i_train].reshape(-1, 3).to(device)
        train_targets = dataset.targets.reshape(dataset.N, -1, 3)[i_train].reshape(-1, 3).to(device)
        n_total_rays  = train_rays_o.shape[0]
        print(f'done  ({n_total_rays} rays)')
        # Initial shuffle
        shuffle_idx = torch.randperm(n_total_rays, device=device)
        i_batch = 0
    else:
        images = torch.Tensor(images_np).to(device)
        poses  = torch.Tensor(poses_np).to(device)

    N_iters = args.training_iterations + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all training-image rays
            if i_batch + N_rand > n_total_rays:
                print("Shuffle data after an epoch!")
                shuffle_idx = torch.randperm(n_total_rays, device=device)
                i_batch = 0
            batch_idx  = shuffle_idx[i_batch:i_batch + N_rand]
            rays_o_b   = train_rays_o[batch_idx]
            rays_d_b   = train_rays_d[batch_idx]
            target_s   = train_targets[batch_idx]
            batch_rays = torch.stack([rays_o_b, rays_d_b], 0)
            i_batch   += N_rand

        else:
            # Random from one training image
            img_i  = np.random.choice(i_train)
            target = images[img_i]
            pose   = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, torch.Tensor(K), pose)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds  = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o   = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d   = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s   = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)  # noqa: F405
        trans = extras['raw'][..., -1]  # noqa: F841
        loss = img_loss
        psnr = mse2psnr(img_loss)  # noqa: F405

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)  # noqa: F405
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)  # noqa: F405, F841

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        #####           end            #####

        if i%1000==0 and i >= 0:
            infer_path = os.path.join(basedir, expname, 'iter_infer')
            os.makedirs(infer_path, exist_ok=True)

            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses[0:1], hwf, K, args.chunk, render_kwargs_test)

            print('Done rendering')
            infer_rgb = cv2.cvtColor(to8b(rgbs[0]), cv2.COLOR_RGB2BGR)
            infer_disp_norm = to8b(disps[0] / np.max(disps[0]))
            cv2.imwrite(os.path.join(infer_path, f"infer_rgb{i}.png"), infer_rgb)
            cv2.imwrite(os.path.join(infer_path, f"infer_disp_norm{i}.png"), infer_disp_norm)
            cv2.imwrite(os.path.join(infer_path, f"infer_disp{i}.png"), to8b(disps[0]))

            # Apply heatmap colormap
            colored_disps = [plt.cm.inferno(d)[:, :, :3] for d in disps]  # Normalize and apply colormap
            colored_disps_8bit = [np.uint8(c * 255) for c in colored_disps]
            cv2.imwrite(os.path.join(infer_path, f"infer_disp_rgb{i}.png"), colored_disps_8bit[0])
            print("infer result saved")

            # please uncomment the code starts with `wandb` to add your wandb project tracking
            # Log images to W&B
            # wandb.log({
            #     f"Infer RGB (iter {i})": wandb.Image(infer_rgb, caption=f"Infer RGB Iter {i}"),
            #     f"Infer Disp (iter {i})": wandb.Image(to8b(disps[0]), caption=f"Infer Disp Iter {i}"),
            #     f"Infer Disp Norm (iter {i})": wandb.Image(infer_disp_norm, caption=f"Infer Disp Norm Iter {i}"),
            #     f"Infer Disp Heatmap (iter {i})": wandb.Image(colored_disps_8bit[0], caption=f"Infer Disp Heatmap Iter {i}")
            # })
            # print("Saved in WandB")
            logging.info("Rendering inference results...")
            logging.info("Inference results saved to %s", infer_path)

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            logging.info("Saved checkpoints at %s", path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            logging.info("Video saved at %s", moviebase)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            test_poses = torch.Tensor(poses_np[i_test]).to(device)
            print('test poses shape', test_poses.shape)
            with torch.no_grad():
                render_path(test_poses, hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images_np[i_test], savedir=testsavedir)
            print('Saved test set')
            logging.info("Test set saved at %s", testsavedir)

        if i%args.i_print==0:
            log_message = f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}"
            tqdm.write(log_message)
            logging.info(log_message)
            # please uncomment the code starts with `wandb` to add your wandb project tracking
            # wandb.log({"epoch": i, "loss": loss.item(), "PSNR": psnr.item()}, step=N_iters)

        global_step += 1
