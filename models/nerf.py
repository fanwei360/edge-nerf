import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x, step=1, all_steps=2):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        if self.N_freqs == 10:  # xyz PE regularization
            out = [x]
            for freq in self.freq_bands:
                for func in self.funcs:  # sin/cos
                    out += [func(freq*x)]

            out = torch.cat(out, -1)  # [batch_size * nb_bins,3 * (2*L+1)]
            return out
        else:   # dir
            out = [x]
            for freq in self.freq_bands:
                for func in self.funcs:  # sin/cos
                    out += [func(freq * x)]

            return torch.cat(out, -1)  # （B,2*3*N_freqs+3）


class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 D2=4,
                 in_channels_xyz=63, in_channels_dir=27,
                 skips=[4],
                 enable_edge=True):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.D2 = D2
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # self.enable_edge = enable_edge

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)  # setattr(x, 'y', v) is equivalent to ``x.y = v''

        self.feature_linear = nn.Linear(W, W)
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(nn.Linear(W // 2, 3))

        # semantic layers
        for i in range(D2):
            if i == 0:
                layer = nn.Sequential(nn.Linear(in_channels_xyz, W), nn.ReLU(True), nn.BatchNorm1d(W))
            else:
                layer = nn.Sequential(nn.Linear(W, W), nn.ReLU(True), nn.BatchNorm1d(W))
            setattr(self, f"xyz_encoding_e_density_{i + 1}", layer)
        self.e_sigma = nn.Sequential(nn.Linear(W, 1), nn.Sigmoid())
        self.e_feature_linear = nn.Linear(W, W)
        self.edge = nn.Sequential(nn.Linear(W+in_channels_dir, W // 2),
                                  nn.ReLU(True),
                                  nn.Linear(W // 2, 1),
                                  nn.Sigmoid())

        for name, layer in self.named_children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x, out_typ, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)


        # ===========================================================
        if out_typ == 'all':
            sigma = self.sigma(xyz_)
            # sigma = self.sigmoid(sigma)
            if sigma_only:
                return sigma
            # add dir
            xyz_encoding_feature = self.feature_linear(xyz_)
            dir_encoding_input = torch.cat([xyz_encoding_feature, input_dir], -1)
            dir_encoding = self.dir_encoding(dir_encoding_input)    # 128
            rgb = self.rgb(dir_encoding)    # 1

            e_xyz = input_xyz
            for i in range(self.D2):
                e_xyz = getattr(self, f"xyz_encoding_e_density_{i+1}")(e_xyz)
            e_density = self.e_sigma(e_xyz)
            e_xyz_encoding_feature = self.e_feature_linear(e_xyz)
            e_dir_encoding_input = torch.cat([e_xyz_encoding_feature, input_dir], -1)
            edge_gray = self.edge(e_dir_encoding_input)
            out = torch.cat([rgb, sigma, edge_gray, e_density], -1)    # 3+1+1+1

        elif out_typ == 'edge':
            e_xyz = input_xyz
            sigma = self.sigma(xyz_)
            for i in range(self.D2):
                e_xyz = getattr(self, f"xyz_encoding_e_density_{i + 1}")(e_xyz)
            e_density = self.e_sigma(e_xyz)
            e_xyz_encoding_feature = self.e_feature_linear(e_xyz)
            e_dir_encoding_input = torch.cat([e_xyz_encoding_feature, input_dir], -1)
            edge_gray = self.edge(e_dir_encoding_input)
            out = torch.cat([edge_gray, e_density, sigma], -1)  # 1+1

        else:   # out_typ = 'rgb'
            sigma = self.sigma(xyz_)
            # sigma = self.sigmoid(sigma)
            if sigma_only:
                return sigma
            # add dir
            xyz_encoding_feature = self.feature_linear(xyz_)
            dir_encoding_input = torch.cat([xyz_encoding_feature, input_dir], -1)
            dir_encoding = self.dir_encoding(dir_encoding_input)  # 128
            rgb = self.rgb(dir_encoding)  # 1
            out = torch.cat([rgb, sigma], -1)   # 3+1
        return out


