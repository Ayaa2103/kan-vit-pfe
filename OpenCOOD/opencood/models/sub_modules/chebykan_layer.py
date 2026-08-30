# ChebyKANLayer, vendored from https://github.com/SynodicMonth/ChebyKAN
# (MIT license), not available as a pip package. Kolmogorov-Arnold layer
# using Chebyshev polynomials of the first kind as the learnable per-edge
# basis functions, instead of the original KAN's B-splines -- much cheaper
# to evaluate (no grid/knot bookkeeping), which is why it was chosen here
# over pykan/efficient-kan for a 4GB-VRAM GPU.
import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Chebyshev polynomials are defined on [-1, 1], so normalize x there first.
        x = torch.tanh(x)
        # (batch, inputdim) -> (batch, inputdim, degree+1)
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        # T_n(cos(theta)) = cos(n * theta), theta = acos(x)
        x = x.acos()
        x = x * self.arange
        x = x.cos()
        # Chebyshev interpolation: contract with the learnable coefficients
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        y = y.view(-1, self.outdim)
        return y


class KANFeedForward(nn.Module):
    """
    Drop-in replacement for opencood.models.sub_modules.base_transformer's
    FeedForward, matching its (..., dim) -> (..., dim) shape contract.
    Two chained ChebyKANLayers (dim -> hidden_dim -> dim) mirror the
    original's two nn.Linear layers, but with no activation function
    between them: unlike a Linear layer, each KAN layer already applies
    its own learnable nonlinearity per edge, so inserting e.g. GELU
    between two KAN layers would be redundant rather than necessary.

    Day 5 fix: ChebyKANLayer's intermediate tensor is
    (n_tokens, dim, degree+1). Applied directly to a V2X-ViT spatial
    feature map, n_tokens = B*L*H*W (measured 84,480 for the mini-sample:
    B=2, L=5 agents, H=48, W=176), which blew up to 22.87GB peak GPU
    memory on a 4GB card. The root cause is n_tokens, not the KAN's
    per-connection expressiveness, so pool_size>1 average-pools the
    spatial map (H,W) down before the KAN layers and nearest-upsamples
    the result back afterward -- this cuts n_tokens by pool_size**2
    without touching degree/dim, keeping the KAN itself at full fidelity
    per (pooled) location. Only used when the input is the (B,L,H,W,dim)
    tensor V2XTEncoderKAN passes in; falls back to the plain flatten path
    otherwise (or when pool_size=1, e.g. for standalone/unit-test use).
    """

    def __init__(self, dim, hidden_dim, degree=4, dropout=0., pool_size=1):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size
        self.kan1 = ChebyKANLayer(dim, hidden_dim, degree)
        self.drop1 = nn.Dropout(dropout)
        self.kan2 = ChebyKANLayer(hidden_dim, dim, degree)
        self.drop2 = nn.Dropout(dropout)
        if pool_size > 1:
            self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size,
                                     ceil_mode=True)

    def _kan_forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.dim)
        x = self.kan1(x)
        x = self.drop1(x)
        x = self.kan2(x)
        x = self.drop2(x)
        x = x.reshape(*orig_shape[:-1], self.dim)
        return x

    def forward(self, x):
        if self.pool_size <= 1 or x.dim() != 5:
            return self._kan_forward(x)

        # x: (B, L, H, W, dim) -- one spatial BEV feature map per agent.
        B, L, H, W, C = x.shape
        x_bl = x.reshape(B * L, H, W, C).permute(0, 3, 1, 2)  # (B*L, C, H, W)
        x_pooled = self.pool(x_bl)  # (B*L, C, H', W')

        x_kan_in = x_pooled.permute(0, 2, 3, 1)  # (B*L, H', W', C)
        x_kan_out = self._kan_forward(x_kan_in)  # (B*L, H', W', C)

        x_kan_out = x_kan_out.permute(0, 3, 1, 2)  # (B*L, C, H', W')
        x_up = nn.functional.interpolate(x_kan_out, size=(H, W), mode='nearest')
        x_up = x_up.permute(0, 2, 3, 1).reshape(B, L, H, W, C)
        return x_up
