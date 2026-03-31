"""Parameter sharing strategies for low-dimensional evolutionary search."""

import numpy as np
import torch
import torch.nn as nn


class RandomProjection:
    """Random projection from k-dimensional latent space to full parameter space.

    The latent vector has length ``k + 1``: the first ``k`` components map through
    a fixed random matrix ``P``; the last component is a scalar **s** that scales
    the projected delta: ``delta = s * (P @ z)``.  Initial ``s`` is 1 so that with
    ``z = 0`` the delta is still zero (``P @ 0 = 0``).
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.projections = np.random.randn(self.N, self.k) / np.sqrt(self.k)
        self.num_dims = self.k + 1
        self.z0 = self._init_z0()

    def _init_z0(self):
        z0 = np.zeros(self.num_dims)
        z0[-1] = 1.0
        return z0

    def expand(self, z):
        """Map latent vector z to full parameter space via random projection."""
        if not isinstance(z, np.ndarray):
            z = np.asarray(z, dtype=np.float64)
        z_lat = z[: self.k]
        s = float(z[self.k])
        x = self.projections @ z_lat
        return s * x

    def process(self, x, alpha=1.0):
        """Convert expanded parameters to model-compatible form."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        x = x.to(self.device)
        return alpha * x

    def load_parameters_to_model(self, parameters):
        """Set model weights from a flat array."""
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        parameters = parameters.to(self.device)
        torch.nn.utils.vector_to_parameters(parameters, self.model.parameters())

    def forward(self, z):
        """Full forward: z -> expand -> process -> theta."""
        x = self.expand(z)
        return self.process(x)


class LayerwiseRandomProjection:
    """Layer-wise random projection with direct bias evolution.

    Each non-bias parameter tensor gets its own projection matrix
    ``P_l in R^{n_l x k}``, so latent dimensionality is::

        num_dims = k * L + sum(bias_sizes)

    where ``L`` is the number of projected (non-bias) parameter tensors.
    Bias tensors are appended directly to ``z`` (no projection).
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        self.projections = []
        total_dims = 0
        n_projected_layers = 0
        n_bias_dims = 0

        for name, param in self.model.named_parameters():
            shape = param.shape
            size = int(np.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

            if name.endswith(".bias"):
                dims = size
                self.layer_info.append(
                    {
                        "type": "direct",
                        "name": name,
                        "n": size,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                    }
                )
                total_dims += dims
                n_bias_dims += dims
            else:
                P = np.random.randn(size, self.k) / np.sqrt(self.k)
                proj_idx = len(self.projections)
                self.projections.append(P)
                dims = self.k
                self.layer_info.append(
                    {
                        "type": "proj",
                        "name": name,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                        "proj_idx": proj_idx,
                        "k": self.k,
                    }
                )
                total_dims += dims
                n_projected_layers += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LayerwiseRandomProj: L={n_projected_layers}, k={self.k}, "
            f"bias_dims={n_bias_dims} | z dim = {self.num_dims} | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def expand(self, z):
        """Expand layer-wise latent vector to full parameter-space delta."""
        if not isinstance(z, np.ndarray):
            z = np.asarray(z, dtype=np.float64)

        reconstructed = []
        for info in self.layer_info:
            offset = info["offset"]
            dims = info["dims"]
            z_layer = z[offset : offset + dims]

            if info["type"] == "proj":
                P = self.projections[info["proj_idx"]]
                reconstructed.append(P @ z_layer)
            else:
                reconstructed.append(z_layer.copy())

        return np.concatenate(reconstructed)

    def process(self, x, alpha=1.0):
        """Convert expanded parameters to model-compatible form."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        x = x.to(self.device)
        return alpha * x

    def load_parameters_to_model(self, parameters):
        """Set model weights from a flat array."""
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        parameters = parameters.to(self.device)
        torch.nn.utils.vector_to_parameters(parameters, self.model.parameters())

    def forward(self, z):
        """Full forward: z -> expand -> process -> theta."""
        x = self.expand(z)
        return self.process(x)


class LayerwiseRandomBlocking:
    """Layer-wise random blocking: k latent scalars per weight tensor.

    Each non-bias parameter tensor (flattened to length ``n_l``) is partitioned
    into ``k`` blocks by randomly assigning every weight index to one of ``k``
    subspace indices via a random assignment array ``assignment`` of length ``n_l``::

        delta[i] = z_l[assignment[i]]     for i in 0 … n_l-1

    All weights assigned to block ``j`` share the value ``z_l[j]``.

    If ``n_l < k``, that tensor is evolved **directly** (like bias): ``z_l`` has
    length ``n_l`` and ``delta = z_l``.

    Biases are always direct. Total latent dimension is
    ``sum_l min(k, n_l) + sum(bias_sizes)`` over weight tensors ``l`` and biases.
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_blocking_layers = 0
        n_bias_dims = 0
        n_direct_weight_dims = 0

        for name, param in self.model.named_parameters():
            shape = param.shape
            size = int(np.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

            if name.endswith(".bias"):
                dims = size
                self.layer_info.append(
                    {
                        "type": "direct",
                        "name": name,
                        "n": size,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                    }
                )
                total_dims += dims
                n_bias_dims += dims
            else:
                if self.k > size:
                    self.layer_info.append(
                        {
                            "type": "direct",
                            "name": name,
                            "n": size,
                            "dims": size,
                            "offset": total_dims,
                            "size": size,
                        }
                    )
                    total_dims += size
                    n_direct_weight_dims += size
                else:
                    base = np.tile(np.arange(self.k), np.ceil(size / self.k).astype(int))[:size]
                    assignment = rng.permutation(base).astype(np.int64)
                    self.layer_info.append(
                        {
                            "type": "rand_blocking",
                            "name": name,
                            "dims": self.k,
                            "offset": total_dims,
                            "size": size,
                            "assignment": assignment,
                        }
                    )
                    total_dims += self.k
                    n_blocking_layers += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LayerwiseRandomBlocking: blocking_layers={n_blocking_layers}, k={self.k}, "
            f"bias_dims={n_bias_dims}, direct_weight_dims={n_direct_weight_dims} | "
            f"z dim = {self.num_dims} | model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def expand(self, z):
        """Expand latent vector to full parameter-space delta via random index ties."""
        if not isinstance(z, np.ndarray):
            z = np.asarray(z, dtype=np.float64)

        reconstructed = []
        for info in self.layer_info:
            offset = info["offset"]
            dims = info["dims"]
            z_layer = z[offset : offset + dims]

            if info["type"] == "rand_blocking":
                reconstructed.append(z_layer[info["assignment"]])
            else:
                reconstructed.append(z_layer.copy())

        return np.concatenate(reconstructed)

    def process(self, x, alpha=1.0):
        """Convert expanded parameters to model-compatible form."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        x = x.to(self.device)
        return alpha * x

    def load_parameters_to_model(self, parameters):
        """Set model weights from a flat array."""
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        parameters = parameters.to(self.device)
        torch.nn.utils.vector_to_parameters(parameters, self.model.parameters())

    def forward(self, z):
        """Full forward: z -> expand -> process -> theta."""
        x = self.expand(z)
        return self.process(x)


class LayerwiseScaledRandomProjection:
    """Layer-wise random projection with per-layer alpha and direct biases.

    Each non-bias tensor ``l`` has a projection matrix ``P_l in R^{n_l x k}`` and
    a dedicated scalar ``alpha_l``:

        delta_l = alpha_l * (P_l @ z_l)

    Latent dimensionality is:

        num_dims = (k + 1) * L + sum(bias_sizes)

    where ``L`` is the number of projected (non-bias) parameter tensors.
    Bias tensors are evolved directly (no projection).
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        self.projections = []
        total_dims = 0
        n_projected_layers = 0
        n_bias_dims = 0

        for name, param in self.model.named_parameters():
            shape = param.shape
            size = int(np.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

            if name.endswith(".bias"):
                dims = size
                self.layer_info.append(
                    {
                        "type": "direct",
                        "name": name,
                        "n": size,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                    }
                )
                total_dims += dims
                n_bias_dims += dims
            else:
                P = np.random.randn(size, self.k) / np.sqrt(self.k)
                proj_idx = len(self.projections)
                self.projections.append(P)
                dims = self.k + 1  # k latent coords + 1 alpha scale
                self.layer_info.append(
                    {
                        "type": "proj_scaled",
                        "name": name,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                        "proj_idx": proj_idx,
                        "k": self.k,
                    }
                )
                total_dims += dims
                n_projected_layers += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LayerwiseScaledRandomProj: L={n_projected_layers}, k={self.k}, "
            f"alpha_dims={n_projected_layers}, bias_dims={n_bias_dims} | "
            f"z dim = {self.num_dims} | model params = {self.N}"
        )

    def _init_z0(self):
        z0 = np.zeros(self.num_dims)
        for info in self.layer_info:
            if info["type"] == "proj_scaled":
                z0[info["offset"] + self.k] = 1.0
        return z0

    def expand(self, z):
        """Expand latent vector to full parameter-space delta."""
        if not isinstance(z, np.ndarray):
            z = np.asarray(z, dtype=np.float64)

        reconstructed = []
        for info in self.layer_info:
            offset = info["offset"]
            dims = info["dims"]
            z_layer = z[offset : offset + dims]

            if info["type"] == "proj_scaled":
                z_lat = z_layer[: self.k]
                alpha = float(z_layer[self.k])
                P = self.projections[info["proj_idx"]]
                reconstructed.append(alpha * (P @ z_lat))
            else:
                reconstructed.append(z_layer.copy())

        return np.concatenate(reconstructed)

    def process(self, x, alpha=1.0):
        """Convert expanded parameters to model-compatible form."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        x = x.to(self.device)
        return alpha * x

    def load_parameters_to_model(self, parameters):
        """Set model weights from a flat array."""
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        parameters = parameters.to(self.device)
        torch.nn.utils.vector_to_parameters(parameters, self.model.parameters())

    def forward(self, z):
        """Full forward: z -> expand -> process -> theta."""
        x = self.expand(z)
        return self.process(x)




class FlattenLoRA:
    """LoRA-style low-rank decomposition over flattened model parameters."""

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.r = r
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        dims_conv = 0
        dims_linear = 0
        dims_other = 0
        module_for_param = _build_param_module_map(model)

        for name, param in model.named_parameters():
            shape = param.shape
            self.param_shapes.append(shape)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if len(shape) == 2:
                m, n = shape
                dims = m * r + n * r
                self.layer_info.append({'type': '2d', 'm': m, 'n': n, 'dims': dims, 'offset': total_dims})
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 1:
                n = shape[0]
                dims = n
                self.layer_info.append({'type': '1d', 'n': n, 'dims': dims, 'offset': total_dims})
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 4:
                out_ch, in_ch, h, w = shape
                m, n = out_ch, in_ch * h * w
                dims = m * r + n * r
                self.layer_info.append({
                    'type': '4d', 'm': m, 'n': n, 'original_shape': shape,
                    'dims': dims, 'offset': total_dims
                })
                total_dims += dims
                if is_conv2d:
                    dims_conv += dims
                else:
                    dims_other += dims
            else:
                n = int(np.prod(shape))
                dims = n
                self.layer_info.append({'type': 'other', 'n': n, 'original_shape': shape, 'dims': dims, 'offset': total_dims})
                total_dims += dims
                dims_other += dims

            self.param_sizes.append(int(np.prod(shape)))

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        extra = f", other={dims_other}" if dims_other else ""
        print(
            f"FlattenLoRA: z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}{extra}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def expand(self, z):
        """Expand low-rank vector z to full parameter space."""
        if not isinstance(z, np.ndarray):
            z = np.array(z)

        reconstructed_params = []
        for layer_info in self.layer_info:
            offset = layer_info['offset']
            dims = layer_info['dims']
            z_layer = z[offset:offset + dims]

            if layer_info['type'] == '2d':
                m, n, r = layer_info['m'], layer_info['n'], self.r
                A_flat = z_layer[:m * r]
                B_flat = z_layer[m * r:]
                A = A_flat.reshape(m, r)
                B = B_flat.reshape(n, r)
                W = (A @ B.T) * (1.0 / np.sqrt(r))
                reconstructed_params.append(W.flatten())
            elif layer_info['type'] == '1d':
                reconstructed_params.append(z_layer[: layer_info['n']])
            elif layer_info['type'] == '4d':
                m, n = layer_info['m'], layer_info['n']
                original_shape = layer_info['original_shape']
                r = self.r
                A_flat = z_layer[:m * r]
                B_flat = z_layer[m * r:]
                A = A_flat.reshape(m, r)
                B = B_flat.reshape(n, r)
                W_2d = A @ B.T
                W = W_2d.reshape(original_shape) * (1.0 / np.sqrt(r))
                reconstructed_params.append(W.flatten())
            elif layer_info['type'] == 'other':
                n = layer_info['n']
                reconstructed_params.append(z_layer[:n] * (1.0 / np.sqrt(n)))

        return np.concatenate(reconstructed_params)

    def process(self, x, alpha=1.0):
        """Convert to tensor."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        x = x.to(self.device)
        return alpha * x

    def load_parameters_to_model(self, parameters):
        """Set model weights from flat array."""
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        parameters = parameters.to(self.device)
        torch.nn.utils.vector_to_parameters(parameters, self.model.parameters())

    def forward(self, z):
        """Full forward pass."""
        x = self.expand(z)
        return self.process(x)


class DictLoRA:
    """
    Dictionary-based LoRA for ConvNets.
    For Conv2d: ΔW[o,i,:,:] = Σ_m A[o,m]·B[i,m]·D[m,:,:]
    For Linear/bias: standard LoRA / direct.
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.r = r
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        base_offset = 0
        total_dims = 0
        dims_conv = 0
        dims_linear = 0
        dims_other = 0
        module_for_param = _build_param_module_map(model)

        for name, param in model.named_parameters():
            shape = param.shape
            self.param_shapes.append(shape)
            size = int(np.prod(shape))
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if len(shape) == 2:
                m, n = shape
                dims = m * r + n * r
                self.layer_info.append({
                    'type': '2d', 'm': m, 'n': n, 'dims': dims, 'offset': total_dims,
                    'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 1:
                n = shape[0]
                dims = n
                self.layer_info.append({
                    'type': '1d', 'n': n, 'dims': dims, 'offset': total_dims,
                    'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 4:
                out_ch, in_ch, kh, kw = shape
                M = self.r
                dims = M * (out_ch + in_ch + kh * kw)
                self.layer_info.append({
                    'type': '4d_dict',
                    'Cout': out_ch, 'Cin': in_ch, 'kh': kh, 'kw': kw,
                    'M': M, 'dims': dims, 'offset': total_dims,
                    'original_shape': shape, 'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                if is_conv2d:
                    dims_conv += dims
                else:
                    dims_other += dims
            else:
                n = size
                dims = n
                self.layer_info.append({
                    'type': 'other', 'n': n, 'original_shape': shape, 'dims': dims,
                    'offset': total_dims, 'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                dims_other += dims

            base_offset += size

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        extra = f", other={dims_other}" if dims_other else ""
        print(
            f"DictLoRA: z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}{extra}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def _expand_4d_dict(self, z_layer, Cout, Cin, kh, kw, M):
        """Compute ΔW from DictLoRA factors A, B, D."""
        A_flat = z_layer[:Cout * M]
        B_flat = z_layer[Cout * M:Cout * M + Cin * M]
        D_flat = z_layer[Cout * M + Cin * M:]
        A = A_flat.reshape(Cout, M)
        B = B_flat.reshape(Cin, M)
        D = D_flat.reshape(M, kh, kw)
        delta_W = np.einsum('om,im,mhw->oihw', A, B, D)
        return delta_W * (1.0 / np.sqrt(M))

    def expand(self, z):
        """Expand latent vector to full parameter space (deltas for 4d_dict)."""
        if not isinstance(z, np.ndarray):
            z = np.array(z)

        reconstructed = []
        for layer_info in self.layer_info:
            offset = layer_info['offset']
            dims = layer_info['dims']
            z_layer = z[offset:offset + dims]

            if layer_info['type'] == '2d':
                m, n, r = layer_info['m'], layer_info['n'], self.r
                A_flat = z_layer[:m * r]
                B_flat = z_layer[m * r:]
                A = A_flat.reshape(m, r)
                B = B_flat.reshape(n, r)
                W = (A @ B.T) * (1.0 / np.sqrt(r))
                reconstructed.append(W.flatten())
            elif layer_info['type'] == '1d':
                reconstructed.append(z_layer[: layer_info['n']])
            elif layer_info['type'] == '4d_dict':
                delta_W = self._expand_4d_dict(
                    z_layer,
                    layer_info['Cout'], layer_info['Cin'],
                    layer_info['kh'], layer_info['kw'],
                    layer_info['M'],
                )
                reconstructed.append(delta_W.flatten())
            elif layer_info['type'] == 'other':
                n = layer_info['n']
                reconstructed.append(z_layer[:n] * (1.0 / np.sqrt(n)))

        return np.concatenate(reconstructed)

    def process(self, x, alpha=1.0):
        """Return delta only; caller adds base_params."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        x = x.to(self.device)
        return alpha * x

    def load_parameters_to_model(self, parameters):
        """Set model weights from flat array."""
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        parameters = parameters.to(self.device)
        torch.nn.utils.vector_to_parameters(parameters, self.model.parameters())

    def forward(self, z):
        """Full forward: returns delta to be added to base_params."""
        x = self.expand(z)
        return self.process(x)


def _build_param_module_map(model):
    """Map each parameter's full name to its immediate parent module type."""
    mapping = {}
    for mod_name, mod in model.named_modules():
        for pname, _ in mod.named_parameters(recurse=False):
            full_name = f"{mod_name}.{pname}" if mod_name else pname
            mapping[full_name] = type(mod)
    return mapping


class LinearOnlyLoRA:
    """LoRA exclusively on ``nn.Linear`` layers; everything else frozen.

    Conv2d, BatchNorm, LayerNorm, and all other parameters produce zero delta.
    Linear weights get rank-*r* LoRA; Linear biases are evolved directly.

    Ideal for architectures where class-discriminative information concentrates
    in linear projections (transformer Q/V/MLP, classifier FC heads).
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_lora = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(np.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_linear and len(shape) == 2:
                m, n = shape
                dims = m * r + n * r
                self.layer_info.append({
                    'type': 'lora_2d', 'name': name, 'm': m, 'n': n,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_lora += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LinearOnlyLoRA: {n_lora} LoRA layers, {n_direct} direct (bias), "
            f"{n_frozen} frozen | z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def expand(self, z):
        """Expand z to a full-length parameter delta (zeros for frozen layers)."""
        if not isinstance(z, np.ndarray):
            z = np.array(z)

        parts = []
        for info in self.layer_info:
            if info['type'] == 'lora_2d':
                offset = info['offset']
                m, n, r = info['m'], info['n'], self.r
                A = z[offset:offset + m * r].reshape(m, r)
                B = z[offset + m * r:offset + m * r + n * r].reshape(n, r)
                parts.append(((A @ B.T) * (1.0 / np.sqrt(r))).flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].copy())
            else:
                parts.append(np.zeros(info['size']))

        return np.concatenate(parts)

    def process(self, x, alpha=1.0):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        return alpha * x.to(self.device)

    def load_parameters_to_model(self, parameters):
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        torch.nn.utils.vector_to_parameters(parameters.to(self.device), self.model.parameters())

    def forward(self, z):
        return self.process(self.expand(z))


class ModulationLoRA:
    """Per-channel multiplicative scaling for Conv2d + LoRA for Linear layers.

    Conv2d weights: each output channel gets a single scale factor gamma.
        ``delta_W[o,:,:,:] = gamma[o] * W_base[o,:,:,:]``
        gamma=0 at init (z0=0) produces zero delta; the base model is preserved.

    Linear weights: standard rank-*r* LoRA.
    Linear biases: evolved directly.
    BatchNorm, LayerNorm, and all other parameters: frozen (zero delta).

    The conv modulation is strictly per-channel, so it cannot destroy the
    spatial filter structure -- it can only amplify or suppress channels.
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_modulated = 0
        n_lora = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(np.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_conv2d and len(shape) == 4:
                Cout = shape[0]
                dims = Cout
                base_w = param.detach().cpu().numpy().copy()
                self.layer_info.append({
                    'type': 'modulation_4d', 'name': name, 'Cout': Cout,
                    'dims': dims, 'offset': total_dims, 'size': size,
                    'shape': shape, 'base_weight': base_w,
                })
                total_dims += dims
                dims_conv += dims
                n_modulated += 1
            elif is_linear and len(shape) == 2:
                m, n = shape
                dims = m * r + n * r
                self.layer_info.append({
                    'type': 'lora_2d', 'name': name, 'm': m, 'n': n,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_lora += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"ModulationLoRA: {n_modulated} modulated conv, {n_lora} LoRA linear, "
            f"{n_direct} direct (bias), {n_frozen} frozen | "
            f"z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def expand(self, z):
        """Expand z to a full-length parameter delta.

        Conv deltas are multiplicative: ``gamma * base_weight``.
        LoRA deltas are additive: ``A @ B^T / sqrt(r)``.
        """
        if not isinstance(z, np.ndarray):
            z = np.array(z)

        parts = []
        for info in self.layer_info:
            if info['type'] == 'modulation_4d':
                offset = info['offset']
                Cout = info['Cout']
                gamma = z[offset:offset + Cout]
                base_w = info['base_weight']
                delta = gamma.reshape(Cout, 1, 1, 1) * base_w
                parts.append(delta.flatten())
            elif info['type'] == 'lora_2d':
                offset = info['offset']
                m, n, r = info['m'], info['n'], self.r
                A = z[offset:offset + m * r].reshape(m, r)
                B = z[offset + m * r:offset + m * r + n * r].reshape(n, r)
                parts.append(((A @ B.T) * (1.0 / np.sqrt(r))).flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].copy())
            else:
                parts.append(np.zeros(info['size']))

        return np.concatenate(parts)

    def process(self, x, alpha=1.0):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        return alpha * x.to(self.device)

    def load_parameters_to_model(self, parameters):
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        torch.nn.utils.vector_to_parameters(parameters.to(self.device), self.model.parameters())

    def forward(self, z):
        return self.process(self.expand(z))


class SpectralLoRA:
    """SVD spectral modulation for Conv2d + LoRA for Linear layers.

    Each Conv2d weight is reshaped to 2D and decomposed via SVD at init::

        W_2d = U @ diag(sigma) @ V^T

    The frozen singular vectors ``U[:, :k]`` and ``V[:, :k]`` define the
    eigenbasis; evolution modulates the top-*k* singular values only::

        delta_W_2d = U[:, :k] @ diag(z_layer) @ V[:, :k]^T

    The mapping z -> delta is **linear**: each z_i independently controls
    one orthogonal mode of the weight matrix.  Spatial filter structure is
    completely preserved because U and V are frozen from the base model.

    Linear weights: standard rank-*r* LoRA (bilinear).
    Linear biases: evolved directly.
    BatchNorm, LayerNorm, and all other parameters: frozen (zero delta).

    ``lora_rank`` controls both the number of SVD modes *k* for conv layers
    and the LoRA rank *r* for linear layers.
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_spectral = 0
        n_lora = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(np.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_conv2d and len(shape) == 4:
                Cout, Cin, kh, kw = shape
                W_2d = param.detach().cpu().numpy().reshape(Cout, Cin * kh * kw)
                full_rank = min(Cout, Cin * kh * kw)
                k = min(r, full_rank)
                U_full, sigma, Vt_full = np.linalg.svd(W_2d, full_matrices=False)
                U_k = U_full[:, :k].copy()
                Vt_k = Vt_full[:k, :].copy()

                self.layer_info.append({
                    'type': 'spectral_4d', 'name': name,
                    'k': k, 'dims': k, 'offset': total_dims, 'size': size,
                    'shape': shape, 'U_k': U_k, 'Vt_k': Vt_k,
                    'sigma_k': sigma[:k].copy(),
                })
                total_dims += k
                dims_conv += k
                n_spectral += 1
            elif is_linear and len(shape) == 2:
                m, n = shape
                dims = m * r + n * r
                self.layer_info.append({
                    'type': 'lora_2d', 'name': name, 'm': m, 'n': n,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_lora += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"SpectralLoRA: {n_spectral} spectral conv (k={r}), {n_lora} LoRA linear, "
            f"{n_direct} direct (bias), {n_frozen} frozen | "
            f"z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def expand(self, z):
        """Expand z to a full-length parameter delta.

        Conv deltas via spectral modulation: ``U_k @ diag(z) @ Vt_k``.
        Linear deltas via LoRA: ``A @ B^T / sqrt(r)``.
        """
        if not isinstance(z, np.ndarray):
            z = np.array(z)

        parts = []
        for info in self.layer_info:
            if info['type'] == 'spectral_4d':
                offset = info['offset']
                k = info['k']
                z_sv = z[offset:offset + k]
                delta_2d = (info['U_k'] * z_sv) @ info['Vt_k']
                parts.append(delta_2d.flatten())
            elif info['type'] == 'lora_2d':
                offset = info['offset']
                m, n, r = info['m'], info['n'], self.r
                A = z[offset:offset + m * r].reshape(m, r)
                B = z[offset + m * r:offset + m * r + n * r].reshape(n, r)
                parts.append(((A @ B.T) * (1.0 / np.sqrt(r))).flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].copy())
            else:
                parts.append(np.zeros(info['size']))

        return np.concatenate(parts)

    def process(self, x, alpha=1.0):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        return alpha * x.to(self.device)

    def load_parameters_to_model(self, parameters):
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        torch.nn.utils.vector_to_parameters(parameters.to(self.device), self.model.parameters())

    def forward(self, z):
        return self.process(self.expand(z))


class SpectralAllSVD:
    """SVD spectral modulation for Conv2d and Linear 2D weights (no LoRA).

    Conv2d and ``nn.Linear`` weight matrices are treated the same: each is
    decomposed at init as ``W = U @ diag(sigma) @ V^T``; evolution only
    modulates the top-*k* singular values with frozen *U*, *V*::

        delta_W = U[:, :k] @ diag(z_layer) @ V[:, :k]^T

    Linear biases are evolved directly. BatchNorm, LayerNorm, and all other
    parameters stay frozen (zero delta).

    ``r`` sets *k* = ``min(r, rank(W))`` per layer for both conv and linear
    weights.
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_spectral_conv = 0
        n_spectral_linear = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(np.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_conv2d and len(shape) == 4:
                Cout, Cin, kh, kw = shape
                W_2d = param.detach().cpu().numpy().reshape(Cout, Cin * kh * kw)
                full_rank = min(Cout, Cin * kh * kw)
                k = min(r, full_rank)
                U_full, sigma, Vt_full = np.linalg.svd(W_2d, full_matrices=False)
                U_k = U_full[:, :k].copy()
                Vt_k = Vt_full[:k, :].copy()

                self.layer_info.append({
                    'type': 'spectral_4d', 'name': name,
                    'k': k, 'dims': k, 'offset': total_dims, 'size': size,
                    'shape': shape, 'U_k': U_k, 'Vt_k': Vt_k,
                    'sigma_k': sigma[:k].copy(),
                })
                total_dims += k
                dims_conv += k
                n_spectral_conv += 1
            elif is_linear and len(shape) == 2:
                W_2d = param.detach().cpu().numpy()
                m, n = W_2d.shape
                full_rank = min(m, n)
                k = min(r, full_rank)
                U_full, sigma, Vt_full = np.linalg.svd(W_2d, full_matrices=False)
                U_k = U_full[:, :k].copy()
                Vt_k = Vt_full[:k, :].copy()

                self.layer_info.append({
                    'type': 'spectral_2d', 'name': name,
                    'k': k, 'dims': k, 'offset': total_dims, 'size': size,
                    'shape': shape, 'U_k': U_k, 'Vt_k': Vt_k,
                    'sigma_k': sigma[:k].copy(),
                })
                total_dims += k
                dims_linear += k
                n_spectral_linear += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"SpectralAllSVD: {n_spectral_conv} spectral conv + {n_spectral_linear} spectral linear "
            f"(k<={r}), {n_direct} direct (bias), {n_frozen} frozen | "
            f"z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return np.zeros(self.num_dims)

    def expand(self, z):
        """Deltas via ``U_k @ diag(z) @ Vt_k`` for conv and linear weights."""
        if not isinstance(z, np.ndarray):
            z = np.array(z)

        parts = []
        for info in self.layer_info:
            if info['type'] in ('spectral_4d', 'spectral_2d'):
                offset = info['offset']
                k = info['k']
                z_sv = z[offset:offset + k]
                delta_2d = (info['U_k'] * z_sv) @ info['Vt_k']
                parts.append(delta_2d.flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].copy())
            else:
                parts.append(np.zeros(info['size']))

        return np.concatenate(parts)

    def process(self, x, alpha=1.0):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
        return alpha * x.to(self.device)

    def load_parameters_to_model(self, parameters):
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(parameters).to(self.device).float()
        torch.nn.utils.vector_to_parameters(parameters.to(self.device), self.model.parameters())

    def forward(self, z):
        return self.process(self.expand(z))
