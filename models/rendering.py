import torch
from einops import rearrange, reduce, repeat
__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])


    return samples

def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    def inference(results, model, typ, xyz, z_vals, edge_c, test_time=False, output_edge=True, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                        # (N_rays*N_samples_, embed_dir_channels)
        for i in range(0, B, chunk):
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk], kwargs["batch_nb"], kwargs["all_batch"])  #  3->63
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded_[i:i+chunk]], 1)  # 63+27
            if typ == 'coarse' or 'fine':
                out_typ = 'all'
                out_c = 6
            elif output_edge:
                out_typ = 'edge'
                out_c = 3
            else:
                out_typ = 'rgb'
                out_c = 4

            out_chunks += [model(xyzdir_embedded, out_typ, sigma_only=False)]  # model forward
        out = torch.cat(out_chunks, 0)  # chunk,3(rgb)+1(sigma)+2(classes)+1(edge_density)
        out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=out_c)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        if typ == 'coarse' or 'fine':
            rgbs = torch.sigmoid(out[..., :3])  # (N_rays, N_samples_, 3) 第0个
            sigmas = out[..., 3]  # (N_rays, N_samples_)
            sem_ret = torch.unsqueeze(out[..., 4], dim=-1)
            e_density = out[..., 5]
            beta=0.1
            # results[f'spacial_sigmas_{typ}'] = sigmas.squeeze()
            results[f'spacial_sem_ret_{typ}'] = sem_ret
            results[f'spacial_e_density_{typ}'] = e_density.squeeze()

            # noise = torch.randn_like(sigmas) * noise_std
            # compute alpha by the formula (3), raw NeRF
            noise = torch.randn_like(sigmas) * noise_std
            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)

            alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10],
                                       -1)  # [1, 1-a1, 1-a2, ...]
            weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)
            weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')  # (N_rays), the accumulated opacity along the rays, equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
            e_alphas = alphas * e_density
            e_alphas_shifted = torch.cat([torch.ones_like(e_alphas[:, :1]), 1 - e_alphas + 1e-10],
                                         -1)  # [1, 1-a1, 1-a2, ...]
            e_weights = e_alphas * torch.cumprod(e_alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_) 权
            e_weights_sum = reduce(e_weights, 'n1 n2 -> n1',
                                   'sum')  # (N_rays), the accumulated opacity along the rays, equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
            if test_time:
                e_alphas_ = alphas * e_density * (e_density > 0.2) # filter
                e_alphas_[:, -1] = 1
                e_alphas_shifted_ = torch.cat([torch.ones_like(e_alphas_[:, :1]), 1 - e_alphas_ + 1e-10],
                                              -1)  # [1, 1-a1, 1-a2, ...]
                e_weights_ = e_alphas_ * torch.cumprod(e_alphas_shifted_[:, :-1],
                                                       -1)  # (N_rays, N_samples_)
                edge_density_depth_map = reduce(e_weights_ * z_vals, 'n1 n2 -> n1', 'sum')
                results[f'edge_depth_cut_{typ}'] = edge_density_depth_map
                edge_map_show = reduce(rearrange(e_weights_, 'n1 n2 -> n1 n2 1') * sem_ret, 'n1 n2 c -> n1 c',
                                  'sum')
                results[f'edge_map_cut_{typ}'] = edge_map_show

            results[f'weights_{typ}'] = weights
            results[f'e_weights_{typ}'] = e_weights

            rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * rgbs, 'n1 n2 c -> n1 c',
                             'sum')
            depth_map = reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')
            edge_map = reduce(rearrange(e_weights, 'n1 n2 -> n1 n2 1') * sem_ret, 'n1 n2 c -> n1 c',
                              'sum')
            edge_depth_map = reduce(e_weights * z_vals, 'n1 n2 -> n1', 'sum')
            if white_back:
                rgb_map += 1 - weights_sum.unsqueeze(1)
            results[f'edge_map_{typ}'] = edge_map
            results[f'edge_depth_{typ}'] = edge_depth_map
            results[f'rgb_{typ}'] = rgb_map
            results[f'depth_{typ}'] = depth_map
            return beta
        elif output_edge:  # for test
            sem_ret = torch.unsqueeze(out[..., 0], dim=-1)
            e_density = out[..., 1]
            sigmas = out[..., -1]
            e_sigmas = sigmas * e_density
            e_sigmas_d = sigmas / (1 + torch.exp(-20*(e_density - 0.6)))

            beta = 0.1
            results[f'spacial_sem_ret_{typ}'] = sem_ret
            results[f'spacial_e_density_{typ}'] = e_density.squeeze()

            noise = torch.randn_like(e_sigmas) * noise_std
            e_alphas = 1 - torch.exp(-deltas * torch.relu(e_sigmas + noise))  # (N_rays, N_samples_)
            e_alphas_shifted = torch.cat([torch.ones_like(e_alphas[:, :1]), 1 - e_alphas + 1e-10],
                                         -1)  # [1, 1-a1, 1-a2, ...]
            e_weights = e_alphas * torch.cumprod(e_alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)
            e_weights_sum = reduce(e_weights, 'n1 n2 -> n1', 'sum')  # (N_rays), the accumulated opacity along the rays

            e_alphas_ = 1 - torch.exp(-deltas * torch.relu(e_sigmas_d + noise))  # (N_rays, N_samples_)
            e_alphas_shifted_ = torch.cat([torch.ones_like(e_alphas_[:, :1]), 1 - e_alphas_ + 1e-10],
                                          -1)  # [1, 1-a1, 1-a2, ...]
            e_weights_ = e_alphas_ * torch.cumprod(e_alphas_shifted_[:, :-1], -1)  # (N_rays, N_samples_)
            edge_density_depth_map = reduce(e_weights_ * z_vals, 'n1 n2 -> n1', 'sum')
            results[f'edge_depth_cut_{typ}'] = edge_density_depth_map
            if test_time:
                edge_map_show = reduce(rearrange(e_weights_, 'n1 n2 -> n1 n2 1') * sem_ret, 'n1 n2 c -> n1 c',
                                  'sum')
                results[f'edge_map_cut_{typ}'] = edge_map_show

            if test_time and typ == 'coarse' and 'fine' in models:
                return

            edge_map = reduce(rearrange(e_weights, 'n1 n2 -> n1 n2 1') * sem_ret, 'n1 n2 c -> n1 c', 'sum')
            edge_depth_map = reduce(e_weights * z_vals, 'n1 n2 -> n1', 'sum')
            results[f'edge_map_{typ}'] = edge_map
            results[f'edge_depth_{typ}'] = edge_depth_map

            return beta  # test
        else:
            rgbs = torch.sigmoid(out[..., :3])  # (N_rays, N_samples_, 3) 第0个
            sigmas = out[..., 3]  # (N_rays, N_samples_)

            # compute alpha by the formula (3), raw NeRF
            noise = torch.randn_like(sigmas) * noise_std
            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
            alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10],
                                       -1)  # [1, 1-a1, 1-a2, ...]
            weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)
            weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')  # (N_rays), the accumulated opacity along the rays
            if test_time and typ == 'coarse' and 'fine' in models:
                return
            rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * rgbs, 'n1 n2 c -> n1 c', 'sum')
            depth_map = reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')

            if white_back:
                rgb_map += 1 - weights_sum.unsqueeze(1)
            results[f'rgb_{typ}'] = rgb_map
            results[f'depth_{typ}'] = depth_map

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    edge_c = rays[:, -1]
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))  # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')  # (chunk*N_importance*3)
    # print("xyz_coarse.shape:", xyz_coarse.shape)
    results = {}
    beta = inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, edge_c, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: , :-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
        inference(results, models['fine'], 'fine', xyz_fine, z_vals, edge_c, test_time, output_edge=True, **kwargs)

    return results, beta
