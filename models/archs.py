"""
Model architectures for Geometric Color Grading.

- NILUT: Residual MLP baseline.
- SIREN: Periodic-activation INR baseline.
- GeometricNiLUT: 7-DoF (Scale + Rotor + Translation). Enforces conformality.
- GeometricAffineNiLUT: 12-DoF (Full Affine + Translation). Allows shear/stretching.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NILUT(nn.Module):
    """Simple residual coordinate-based neural network for fitting 3D LUTs"""
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=3, res=True):
        super().__init__()
        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity):
        output = self.net(intensity)
        if self.res:
            output = output + intensity
            output = torch.clamp(output, 0., 1.)
        return output, intensity


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class SIREN(nn.Module):
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3, out_features=3, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        return self.net(coords)


# --- Geometric Algebra Helper Functions (Stable Implementation) ---

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (aw*bw - ax*bx - ay*by - az*bz,
         aw*bx + ax*bw + ay*bz - az*by,
         aw*by - ax*bz + ay*bw + az*bx,
         aw*bz + ax*by - ay*bx + az*bw), dim=-1)

def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply rotation: v' = q * (0,v) * q_conj"""
    zeros = torch.zeros_like(v[..., :1])
    vq = torch.cat((zeros, v), dim=-1)
    return _quat_mul(_quat_mul(q, vq), _quat_conj(q))[..., 1:]

def exp_map_bivector_to_quat(b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Stable Exponential Map using Sinc approximation.
    b: [..., 3] Bivector (Rotation axis * angle)
    Returns: [..., 4] Unit Quaternion
    """
    theta = torch.linalg.norm(b, dim=-1, keepdim=True)
    w = torch.cos(theta)
    
    # Stable implementation of sin(theta)/theta
    # When theta is small, use Taylor expansion: 1 - theta^2/6
    scale = torch.where(theta < 1e-6, 
                        1.0 - (theta**2 / 6.0), 
                        torch.sin(theta) / theta)
    
    xyz = b * scale
    q = torch.cat((w, xyz), dim=-1)
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)


class GeometricNiLUT(nn.Module):
    """
    [Conformal Model]
    7-DoF: Uniform Scale (1) + Rotation (3) + Translation (3)
    Hypothesis: Color grading is a conformal transformation.
    """
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Output 7: Scale(1) + Trans(3) + Bivector(3)
        self.backbone = SIREN(in_features, hidden_features, hidden_layers, out_features=7, outermost_linear=True)
        
        # Init close to Identity: s=1, t=0, b=0
        last = self.backbone.net[-1]
        with torch.no_grad():
            last.weight.data.uniform_(-1e-4, 1e-4)
            last.bias.data.zero_()
            last.bias.data[0] = float(np.log(np.expm1(1.0))) # softplus(0.54) approx 1.0

    def forward(self, x):
        params = self.backbone(x)
        s = F.softplus(params[..., 0:1]) + self.eps
        t = params[..., 1:4]
        b = params[..., 4:7]
        
        q = exp_map_bivector_to_quat(b, eps=self.eps)
        x_rot = _quat_apply(q, x)
        y = s * x_rot + t
        return torch.clamp(y, 0.0, 1.0), x


class GeometricAffineNiLUT(nn.Module):
    """
    [General Model]
    12-DoF: Full 3x3 Matrix (9) + Translation (3)
    Hypothesis: Color grading requires shear/non-uniform scaling.
    """
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3):
        super().__init__()
        # Output 12: Matrix(9) + Trans(3)
        self.backbone = SIREN(in_features, hidden_features, hidden_layers, out_features=12, outermost_linear=True)
        
        # Init close to Identity Matrix
        last = self.backbone.net[-1]
        with torch.no_grad():
            last.weight.data.uniform_(-1e-4, 1e-4)
            last.bias.data.zero_()
            # Flattened Identity [1,0,0, 0,1,0, 0,0,1]
            last.bias.data[0] = 1.0
            last.bias.data[4] = 1.0
            last.bias.data[8] = 1.0

    def forward(self, x):
        params = self.backbone(x)
        matrix_params = params[..., :9].view(-1, 3, 3)
        t = params[..., 9:]
        
        # y = A * x + t
        # Unsqueeze x for matmul: [B, 3] -> [B, 3, 1]
        y = torch.matmul(matrix_params, x.unsqueeze(-1)).squeeze(-1) + t
        return torch.clamp(y, 0.0, 1.0), x
