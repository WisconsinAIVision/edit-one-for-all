# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import math
import os
import random
import sys
import time
import traceback
from socket import has_dualstack_ipv6

import dnnlib
import legacy  # pylint: disable=import-error
import matplotlib.cm
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch_utils.ops import upfirdn2d
from tqdm import tqdm

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return input_image_array

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self._dtype         = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args)
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            # res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                if 'stylegan2' in pkl:
                    from training.networks_stylegan2 import Generator
                elif 'stylegan3' in pkl:
                    from training.networks_stylegan3 import Generator
                elif 'stylegan_human' in pkl:
                    from stylegan_human.training_scripts.sg2.training.networks import Generator
                else:
                    raise NameError('Cannot infer model type from pkl name!')

                print(data[key].init_args)
                print(data[key].init_kwargs)
                if 'stylegan_human' in pkl:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs, square=False, padding=True)
                else:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs)
                net.load_state_dict(data[key].state_dict())
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def init_network(self, res,
        pkl             = None,
        w0_seed         = 0,
        w_load          = None,
        w_plus          = True,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        input_transform = None,
        lr              = 0.001,
        **kwargs
        ):
        # Dig up network details.
        self.pkl = pkl
        G = self.get_network(pkl, 'G_ema').to(self._device)
        self.G = G

        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.
        self.w0_seed = w0_seed
        self.w_load = w_load

        if self.w_load is None:
            # Generate random latents.
            z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        else:
            w = self.w_load.clone().to(self._device)

        self.w0 = w.detach().clone()
        self.w_plus = w_plus
        if w_plus:
            self.w = w.detach()
        else:
            self.w = w[:, 0, :].detach()
        self.w.requires_grad = True
        self.w_optim = torch.optim.AdamW([self.w], lr=lr)

        self.feat_refs = None
        self.points0_pt = None

    def update_lr(self, lr):

        del self.w_optim
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        print(f'Rebuild optimizer with lr: {lr}')
        print('    Remain feat_refs and points0_pt')

    def _render_drag_impl(self, res,
        points          = [],
        targets         = [],
        mask            = None,
        lambda_mask     = 10,
        reg             = 0,
        feature_idx     = 5,
        r1              = 3,
        r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        **kwargs
    ):
        G = self.G.to(self._device)
        ws = self.w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        if reset:
            self.feat_refs = None
            self.points0_pt = None
        self.points = points

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        # start = time.time()
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
        # print(f'    G time: {time.time() - start}')
        h, w = G.img_resolution, G.img_resolution
        # start = time.time()
        if is_drag:
            X = torch.linspace(0, h, h)
            Y = torch.linspace(0, w, w)
            xx, yy = torch.meshgrid(X, Y)
            feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')
            if self.feat_refs is None:
                self.feat0_resize = F.interpolate(feat[feature_idx].detach(), [h, w], mode='bilinear')
                self.feat_refs = []
                for point in points:
                    py, px = round(point[0]), round(point[1])
                    self.feat_refs.append(self.feat0_resize[:,:,py,px])
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device) # 1, N, 2
            before_points = points
            # Point tracking with feature matching
            with torch.no_grad():
                for j, point in enumerate(points):
                    r = round(r2 / 512 * h)
                    up = max(point[0] - r, 0)
                    down = min(point[0] + r + 1, h)
                    left = max(point[1] - r, 0)
                    right = min(point[1] + r + 1, w)
                    feat_patch = feat_resize[:,:,up:down,left:right]
                    L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                    _, idx = torch.min(L2.view(1,-1), -1)
                    width = right - left
                    point = [idx.item() // width + up, idx.item() % width + left]
                    points[j] = point
            
            res.points = [[point[0], point[1]] for point in points]
            # print(f'    Point tracking time: {time.time() - start}')
            # Motion supervision
            loss_motion = 0
            res.stop = True
            # start_time = time.time()
            for j, point in enumerate(points):
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
                distance_in_pixel = torch.sqrt((direction**2).sum())
                # print('Distance in pixel: ', distance_in_pixel)
                if (torch.linalg.norm(direction) > max(2 / 512 * h, 2)) or (distance_in_pixel.item()<3):
                    res.stop = False
                if torch.linalg.norm(direction) > 1:
                    distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))
                    
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    
                    gridh = (relis+direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs+direction[0]) / (w-1) * 2 - 1
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
                    loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

            loss = loss_motion
            if mask is not None:
                if mask.min() == 0 and mask.max() == 1:
                    mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
                    loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
                    loss += lambda_mask * loss_fix

            loss += reg * F.l1_loss(ws, self.w0)  # latent code regularization
            if not res.stop:
                self.w_optim.zero_grad()
                loss.backward()
                self.w_optim.step()
                res.delta_ws = (self.w - self.w0).detach()
                res.distance_in_pixel = distance_in_pixel
                res.delta_w_star = res.delta_ws
            # print(f'    Loss time: {time.time() - start_time}')
        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        res.image = img
        res.w = ws.detach().cpu()#.numpy()

#----------------------------------------------------------------------------
# Thao's Edit - Apply delta on w2 = w2 + (w1' - w1)

    def _apply_delta(self, res,
        new_ws          = None,
        # reg             = 0,
        # feature_idx     = 5,
        # r1              = 3,
        # r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        **kwargs
    ):
        G = self.G
        if random_seed is not None:
            z = torch.from_numpy(np.random.RandomState(random_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        w0 = w

        if new_ws is not None:
            ws = new_ws.to(self._device) + w0
        else:
            ws = w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        
        ws = torch.cat([ws[:,:6,:], w0[:,6:,:]], dim=1)

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

        h, w = G.img_resolution, G.img_resolution

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        return img

    def _apply_style_clip(self, res,
        new_ws          = None,
        # reg             = 0,
        # feature_idx     = 5,
        # r1              = 3,
        # r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        **kwargs
    ):
        G = self.G
        if random_seed is not None:
            z = torch.from_numpy(np.random.RandomState(random_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        w0 = w
        if new_ws is not None:
            ws = new_ws.to(self._device) + w0
        else:
            ws = w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        
        # ws = torch.cat([ws[:,:6,:], w0[:,6:,:]], dim=1)

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

        h, w = G.img_resolution, G.img_resolution

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        return img

    def _apply_delta_with_alpha(self, res,
        new_ws          = None,
        # reg             = 0,
        # feature_idx     = 5,
        # r1              = 3,
        # r2              = 12,
        random_seed     = 0,
        alpha           = 1,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        return_tensor   = False,
        **kwargs
    ):
        G = self.G
        if random_seed is not None:
            
            z = torch.from_numpy(np.random.RandomState(random_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        w0 = w
        if new_ws is not None:
            ws = new_ws.to(self._device)*alpha + w0
            # wn = new_ws.to(self._device) + self.w0
        else:
            ws = w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        # ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)
        ws = torch.cat([ws[:,:6,:], w0[:,6:,:]], dim=1)
        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

        h, w = G.img_resolution, G.img_resolution
        img_tensor = img
        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        if return_tensor:
            return img_tensor, img, w0
        else:
            return img

    def _apply_delta_2(self, res,
        new_ws          = None,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        return_tensor   = False,
        # alpha_input     =1,
        return_alpha = False,
        style_clip = False,
        **kwargs
    ):
        G = self.G
        if random_seed is not None:
            z = torch.from_numpy(np.random.RandomState(random_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        w0 = w

        if new_ws is not None:
            # print('Alpha Input: ', alpha_input)
            if new_ws.norm() ==0:
                ws = w
            else:
                wn = self.w0 + new_ws
                delta_w = new_ws
                norm_delta_w = delta_w/delta_w.norm()
                alpha = (wn-w0).flatten().dot(norm_delta_w.flatten())
                ws = alpha*norm_delta_w + w0
        else:
            ws = w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        if not style_clip:
            ws = torch.cat([ws[:,:6,:], w0[:,6:,:]], dim=1)
        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
        img_tensor = img
        h, w = G.img_resolution, G.img_resolution

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)

        if return_tensor:
            if return_alpha:
                return img_tensor, img, alpha
            else:
                return img_tensor, img
        else:
            return img

    def optimize_delta_w_2(self, res,
        new_ws          = None,
        random_seed     = 0,
        # alpha           = None,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        return_distance = False,
        return_tensor   = False,
        log_every       = 50,
        log_tb          = True,
        **kwargs
    ): 
        # print('Optimize with alpha: ', alpha)
        from torch.utils.tensorboard import SummaryWriter
        G = self.G
        iteration = 1000
        learning_rate = 0.001
        w_star = torch.zeros_like(new_ws)
        # w_star = new_ws.detach().clone()
        w_star.requires_grad_(True)

        optimizer = torch.optim.AdamW([w_star], lr=learning_rate)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        criteria_mse = torch.nn.MSELoss()
        criteria_l1 = torch.nn.L1Loss(reduce='sum')

        # Set up tensorboard
        pretrained_weight = self.pkl.split('/')[-1].split('.')[0]
        tb_dir = f'./runs/{pretrained_weight}/{random_seed}'
        if os.path.exists(tb_dir):
            import shutil
            shutil.rmtree(tb_dir)
        writer = SummaryWriter(tb_dir)

        for i in tqdm(range(0,iteration)):
            optimizer.zero_grad()

            # Image Rescontruction Loss
            rd_alpha = np.random.uniform(0,1)
            img_gt, _, w0 = self._apply_delta_with_alpha(res, #res
                                        new_ws, #new_ws
                                        random_seed,
                                        # 1,
                                        rd_alpha,
                                        'const', #noise_mode
                                        trunc_psi,  # trunc_psi,
                                        trunc_cutoff,  # trunc_cutoff,
                                        to_pil=True,
                                        return_tensor=True)
            # img_gt = F.interpolate(img_gt, size=(512, 512), mode='bilinear')
            img_pred, pred_rgb, _ = self._apply_delta_with_alpha(res, #res
                                        w_star, #new_ws
                                        random_seed,
                                        # 1,
                                        rd_alpha,
                                        'const', #noise_mode
                                        trunc_psi,  # trunc_psi,
                                        trunc_cutoff,  # trunc_cutoff,
                                        to_pil=True,
                                        return_tensor=True)
            # img_pred = F.interpolate(img_pred, size=(512, 512), mode='bilinear')
            loss_mse = criteria_mse(img_gt, img_pred)

            # Attribute Loss
            rd_alpha = np.random.uniform(0, 1)
            loss_reg = criteria_l1((w0+rd_alpha*w_star).flatten().dot(w_star.flatten()).detach(), (rd_alpha-1)*w_star.norm()**2)
            loss = loss_mse + 0.02*loss_reg

            loss.backward()
            optimizer.step()
            if log_tb:
                writer.add_scalar('Loss/loss_mse', loss_mse, i)
                writer.add_scalar('Loss/loss_reg', loss_reg, i)
                writer.add_scalar('Distance/cosine', cos(w_star.flatten()[None], new_ws.flatten()[None]), i)
                if (i%log_every)==0:
                    img_pred = img_pred * (10 ** (0 / 20))
                    img_pred = (img_pred * 127.5 + 128).clamp(0, 255).to(torch.uint8)#.permute(1, 2, 0)
                    img_gt = img_gt * (10 ** (0 / 20))
                    img_gt = (img_gt * 127.5 + 128).clamp(0, 255).to(torch.uint8)#.permute(1, 2, 0)
                    writer.add_image('img/example', img_pred[0], i)
                    writer.add_image('img/gt', img_gt[0], i)
            if (i % log_every==0):
                os.makedirs(f'./checkpoints/{pretrained_weight}/{random_seed}/', exist_ok=True)
                torch.save(w_star.detach(), f'./checkpoints/{pretrained_weight}/{random_seed}/{i}.pt')
        
        res.delta_w_star = w_star.detach()
        return w_star.detach()

    def optimize_delta_w_batch(self, res,
        new_ws          = None,
        random_seed     = 0,
        # alpha           = None,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        return_distance = False,
        return_tensor   = False,
        log_every       = 50,
        log_tb          = True,
        **kwargs
    ):
        from torch.utils.tensorboard import SummaryWriter
        G = self.G
        iteration = 60
        learning_rate = 0.001
        batch_size = 16

        # w_star = torch.zeros_like(new_ws)
        w_star = new_ws.detach().clone()
        w_star.requires_grad_(True)

        optimizer = torch.optim.AdamW([w_star], lr= learning_rate)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        criteria_mse = torch.nn.MSELoss()
        criteria_l1 = torch.nn.L1Loss()

        # Set up tensorboard
        pretrained_weight = self.pkl.split('/')[-1].split('.')[0]
        tb_dir = f'./runs/{pretrained_weight}/{random_seed}'
        if os.path.exists(tb_dir):
            import shutil
            shutil.rmtree(tb_dir)
        writer = SummaryWriter(tb_dir)

        label = torch.zeros([batch_size, G.c_dim], device=self._device)
        z = torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, 512)).to(self._device, dtype=self._dtype)
        
        for i in tqdm(range(0, iteration)):
            optimizer.zero_grad()

            # Image Loss
            # rd_alpha = torch.rand(batch_size, device= self._device)
            w0_gt = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
            n, w_layers, l_size = w0_gt.shape
            rd_alpha = np.random.uniform(-1, 1, batch_size)
            rd_alpha = torch.from_numpy(rd_alpha).to(self._device)
            rd_alpha = torch.stack([torch.ones(w_layers, l_size, device=self._device)*rd_a for rd_a in rd_alpha])

            w_gt = new_ws.expand(batch_size, -1, -1)*rd_alpha + w0_gt
            w_pred = w_star.expand(batch_size, -1, -1)*rd_alpha + w0_gt
            
            w_gt_input = torch.cat([w_gt[:,:6,:], w0_gt[:,6:,:]], dim=1)
            w_pred_input = torch.cat([w_pred[:,:6,:], w0_gt[:,6:,:]], dim=1)

            img_gt, _ = G(w_gt_input, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
            img_pred, _ = G(w_pred_input, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

            loss_mse = criteria_mse(img_gt, img_pred) #+ loss_mse_test

            # Attribute Loss
            rd_alpha = np.random.uniform(-1, 1, batch_size)
            rd_alpha = torch.from_numpy(rd_alpha).to(self._device)
            rd_alpha = torch.stack([torch.ones(w_layers, l_size, device=self._device)*rd_a for rd_a in rd_alpha])
            dot_product = w_pred.reshape(batch_size, w_layers*l_size)*(w_star.expand(batch_size, -1, -1).reshape(batch_size, w_layers*l_size))
            pred_distance = dot_product.sum(dim=1).detach()
            gt_distance = (w_star.expand(batch_size, -1, -1)*rd_alpha).reshape(batch_size, w_layers*l_size).norm(dim=1)**2
            loss_reg = criteria_l1(pred_distance, gt_distance)
            loss = loss_mse + 0.02*loss_reg
            
            # Total Loss
            # loss = loss_mse + 0.001*loss_reg
            
            loss.backward()
            optimizer.step()

        print('Done')
        res.delta_w_star = w_star#.detach()
        return w_star#.detach()

    def optimize_delta_w_batch_styleclip(self, res,
        new_ws          = None,
        random_seed     = 0,
        # alpha           = None,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        return_distance = False,
        return_tensor   = False,
        log_every       = 50,
        log_tb          = True,
        **kwargs
    ):
        from torch.utils.tensorboard import SummaryWriter
        G = self.G
        iteration = 150
        learning_rate = 0.001
        batch_size = 8

        w_star = torch.zeros_like(new_ws)
        # w_star = new_ws.detach().clone()
        w_star.requires_grad_(True)

        optimizer = torch.optim.AdamW([w_star], lr=learning_rate)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        criteria_mse = torch.nn.MSELoss()
        criteria_l1 = torch.nn.L1Loss()

        # Set up tensorboard
        pretrained_weight = self.pkl.split('/')[-1].split('.')[0]
        tb_dir = f'./runs/{pretrained_weight}/{random_seed}'
        if os.path.exists(tb_dir):
            import shutil
            shutil.rmtree(tb_dir)
        writer = SummaryWriter(tb_dir)

        label = torch.zeros([batch_size, G.c_dim], device=self._device)
        z = torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, 512)).to(self._device, dtype=self._dtype)
        
        for i in tqdm(range(0, iteration)):
            optimizer.zero_grad()

            # Image Loss
            # rd_alpha = torch.rand(batch_size, device= self._device)
            w0_gt = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
            n, w_layers, l_size = w0_gt.shape
            rd_alpha = np.random.uniform(0, 1, batch_size)
            rd_alpha = torch.from_numpy(rd_alpha).to(self._device)
            rd_alpha = torch.stack([torch.ones(w_layers, l_size, device=self._device)*rd_a for rd_a in rd_alpha])

            w_gt = new_ws.expand(batch_size, -1, -1)*rd_alpha + w0_gt
            w_pred = w_star.expand(batch_size, -1, -1)*rd_alpha + w0_gt
            
            # w_gt_input = torch.cat([w_gt[:,:6,:], w0_gt[:,6:,:]], dim=1)
            # w_pred_input = torch.cat([w_pred[:,:6,:], w0_gt[:,6:,:]], dim=1)

            img_gt, _ = G(w_gt, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
            img_pred, _ = G(w_pred, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

            loss_mse = criteria_mse(img_gt, img_pred) #+ loss_mse_test

            # Attribute Loss
            rd_alpha = np.random.uniform(0, 1, batch_size)
            rd_alpha = torch.from_numpy(rd_alpha).to(self._device)
            rd_alpha = torch.stack([torch.ones(w_layers, l_size, device=self._device)*rd_a for rd_a in rd_alpha])
            dot_product = w_pred.reshape(batch_size, w_layers*l_size)*(w_star.expand(batch_size, -1, -1).reshape(batch_size, w_layers*l_size))
            pred_distance = dot_product.sum(dim=1).detach()
            gt_distance = (w_star.expand(batch_size, -1, -1)*rd_alpha).reshape(batch_size, w_layers*l_size).norm(dim=1)**2
            loss_reg = criteria_l1(pred_distance, gt_distance)
            loss = loss_mse + 0.02*loss_reg

            # Total Loss
            # loss = loss_mse + 0.001*loss_reg
            
            loss.backward()
            optimizer.step()
        os.makedirs(f'./checkpoints/{pretrained_weight}/{random_seed}/', exist_ok=True)
        torch.save(w_star.detach(), f'./checkpoints/{pretrained_weight}/{random_seed}/{i}.pt')
        print('Done')
        res.delta_w_star = w_star#.detach()
        return w_star#.detach()

    def optimize_delta_w_batch_without_rd(self, res,
        new_ws          = None,
        random_seed     = 0,
        # alpha           = None,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        return_distance = False,
        return_tensor   = False,
        log_every       = 50,
        log_tb          = True,
        **kwargs
    ):
        from torch.utils.tensorboard import SummaryWriter
        G = self.G
        iteration = 60
        learning_rate = 0.001
        batch_size = 16

        # w_star = torch.zeros_like(new_ws)
        w_star = new_ws.detach().clone()
        w_star.requires_grad_(True)

        optimizer = torch.optim.AdamW([w_star], lr= learning_rate)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        criteria_mse = torch.nn.MSELoss()
        criteria_l1 = torch.nn.L1Loss()

        # Set up tensorboard
        pretrained_weight = self.pkl.split('/')[-1].split('.')[0]
        tb_dir = f'./runs/{pretrained_weight}/{random_seed}'
        if os.path.exists(tb_dir):
            import shutil
            shutil.rmtree(tb_dir)
        writer = SummaryWriter(tb_dir)

        label = torch.zeros([batch_size, G.c_dim], device=self._device)
        z = torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, 512)).to(self._device, dtype=self._dtype)
        
        for i in tqdm(range(0, iteration)):
            optimizer.zero_grad()

            # Image Loss
            # rd_alpha = torch.rand(batch_size, device= self._device)
            w0_gt = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
            n, w_layers, l_size = w0_gt.shape
            rd_alpha = np.ones(batch_size)
            rd_alpha = torch.from_numpy(rd_alpha).to(self._device)
            rd_alpha = torch.stack([torch.ones(w_layers, l_size, device=self._device)*rd_a for rd_a in rd_alpha])

            w_gt = new_ws.expand(batch_size, -1, -1)*rd_alpha + w0_gt
            w_pred = w_star.expand(batch_size, -1, -1)*rd_alpha + w0_gt
            
            w_gt_input = torch.cat([w_gt[:,:6,:], w0_gt[:,6:,:]], dim=1)
            w_pred_input = torch.cat([w_pred[:,:6,:], w0_gt[:,6:,:]], dim=1)

            img_gt, _ = G(w_gt_input, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
            img_pred, _ = G(w_pred_input, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

            loss_mse = criteria_mse(img_gt, img_pred) #+ loss_mse_test

            # Attribute Loss
            rd_alpha = np.ones(batch_size)
            rd_alpha = torch.from_numpy(rd_alpha).to(self._device)
            rd_alpha = torch.stack([torch.ones(w_layers, l_size, device=self._device)*rd_a for rd_a in rd_alpha])
            dot_product = w_pred.reshape(batch_size, w_layers*l_size)*(w_star.expand(batch_size, -1, -1).reshape(batch_size, w_layers*l_size))
            pred_distance = dot_product.sum(dim=1).detach()
            gt_distance = (w_star.expand(batch_size, -1, -1)*rd_alpha).reshape(batch_size, w_layers*l_size).norm(dim=1)**2
            loss_reg = criteria_l1(pred_distance, gt_distance)
            loss = loss_mse# + 0.02*loss_reg
            # loss = loss_reg
            
            # Total Loss
            # loss = loss_mse + 0.001*loss_reg
            
            loss.backward()
            optimizer.step()

        print('Done')
        res.delta_w_star = w_star#.detach()
        return w_star#.detach()

    def _apply_delta_test(self, res,
        new_ws          = None,
        random_seed     = 0,
        alpha           = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        # sel_channels    = 3,
        # base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        reset           = False,
        to_pil          = False,
        return_tensor   = False,
        # alpha_input     =1,
        return_alpha = False,
        style_clip = False,
        **kwargs
    ):
        G = self.G
        if random_seed is not None:
            z = torch.from_numpy(np.random.RandomState(random_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
            alpha = np.random.uniform(-1, 1)
            w = w+new_ws*alpha
        w0 = w

        if new_ws is not None:
            wn = self.w0 + new_ws
            delta_w = new_ws
            norm_delta_w = delta_w/delta_w.norm()
            alpha = (wn-w0).flatten().dot(norm_delta_w.flatten())
            ws = alpha*norm_delta_w + w0
        else:
            ws = w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        if not style_clip:
            ws = torch.cat([ws[:,:6,:], w0[:,6:,:]], dim=1)
        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
        img_tensor = img
        h, w = G.img_resolution, G.img_resolution

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)

        if return_tensor:
            if return_alpha:
                return img_tensor, img, alpha
            else:
                return img_tensor, img
        else:
            return img
