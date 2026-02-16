"""
Model architectures used by this fork.

- NILUT: residual MLP baseline.
- SIREN: periodic-activation INR baseline.
- GeometricNiLUT: SIREN backbone + 7D geometric transform head.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
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
            output = torch.clamp(output, 0.,1.)
        return output, intensity


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
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
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output


# --- Geometric Algebra Helper Functions (Pure PyTorch) ---

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for quaternions.
    a, b: [..., 4] in (w, x, y, z) format
    returns: [..., 4]
    """
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )

def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate. q: [..., 4] -> [..., 4]."""
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply quaternion rotation to 3D vectors using sandwich product: v' = q * (0,v) * q†.
    q: [..., 4] unit quaternion (w,x,y,z)
    v: [..., 3]
    returns: [..., 3]
    """
    zeros = torch.zeros_like(v[..., :1])
    vq = torch.cat((zeros, v), dim=-1)  # [..., 4]
    # q * v * q_conj
    return _quat_mul(_quat_mul(q, vq), _quat_conj(q))[..., 1:]

def exp_map_bivector_to_quat(b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential map from a 3D bivector parameterization to a unit quaternion rotor.
    Given b in R^3 (representing b1*e12 + b2*e23 + b3*e31), define θ = ||b||.
    R = cos(θ) + (b/θ) sin(θ).
    Returned as quaternion (w, x, y, z).
    """
    theta = torch.linalg.norm(b, dim=-1, keepdim=True)  # [..., 1]
    
    # Safe b/theta calculation
    b_over_theta = b / (theta + eps)
    
    w = torch.cos(theta)
    xyz = b_over_theta * torch.sin(theta)
    
    q = torch.cat((w, xyz), dim=-1)
    
    # Numerical safety: normalize to unit length to ensure valid rotation
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)
    return q


class GeometricNiLUT(nn.Module):
    """
    NiLUT variant that predicts geometric transform parameters and applies
    quaternion sandwich rotation using pure PyTorch.

    Network output (7D):
      - s_raw: scaling (R^1) -> s = softplus(s_raw)  (positive)
      - t: translation (R^3)
      - b: rotor bivector (R^3) -> exp_map -> unit quaternion rotor

    Forward:
      x_out = s * (R * x * R†) + t
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        eps: float = 1e-6,
        clamp_output: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.clamp_output = clamp_output

        # Reuse SIREN backbone. 
        # Crucial: outermost_linear=True to get raw unbounded parameters.
        self.backbone = SIREN(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=7, 
            outermost_linear=True,  
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        # Initialize to near-identity transform: s≈1, t≈0, b≈0
        # For s_raw: softplus(s_raw)=1 -> s_raw = log(exp(1)-1) ≈ 0.5413
        last_layer = self.backbone.net[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.data.uniform_(-0.001, 0.001) # Small weights for identity start
                last_layer.bias.data.zero_()
                last_layer.bias.data[0] = float(np.log(np.expm1(1.0)))  # s_raw initial value

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3] RGB in [0,1]
        returns: y [B, 3], x [B, 3]
        """
        # 1. Predict Geometric Parameters
        params = self.backbone(x)  # [B,7] raw
        
        s_raw = params[..., 0:1]
        t = params[..., 1:4]
        b = params[..., 4:7]

        # 2. Map parameters to valid geometric operators
        s = F.softplus(s_raw) + self.eps # Scale must be positive
        q = exp_map_bivector_to_quat(b, eps=self.eps) # Bivector -> Rotor

        # 3. Apply Geometric Transformation: y = s * (R x ~R) + t
        x_rot = _quat_apply(q, x)
        y = s * x_rot + t

        if self.clamp_output:
            y = torch.clamp(y, 0.0, 1.0)
            
        return y, x