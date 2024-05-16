import tinycudann as tcnn
from torch import nn
import torch

def build_tcnn_network(cfg, in_dims, out_dims):
    if cfg.hash_grid_cfg:
        enc_layer = tcnn.Encoding(
            n_input_dims=in_dims,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": cfg.hash_grid_cfg.n_levels,
                "n_features_per_level": cfg.hash_grid_cfg.n_features_per_level,
                "log2_hashmap_size": cfg.hash_grid_cfg.max_hashmap,
                "base_resolution": 16,
                "per_level_scale": 1.447,
            },
        )
        mlp = tcnn.Network(
            n_input_dims=cfg.hash_grid_cfg.n_levels * cfg.hash_grid_cfg.n_features_per_level,
            n_output_dims=out_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": cfg.n_neurons,
                "n_hidden_layers": cfg.n_hidden_layers,
            },
        )
        return nn.Sequential(enc_layer, mlp)
    else:
        return tcnn.Network(
            n_input_dims=in_dims,
            n_output_dims=out_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": cfg.n_neurons,
                "n_hidden_layers": cfg.n_hidden_layers,
            },
        )

def build_nn_network(cfg, in_dims, out_dims):
    modules = []
    in_ch = in_dims
    for i in range(cfg.n_hidden_layers):
        out_ch = cfg.n_neurons
        modules.append(nn.Linear(in_ch, out_ch))
        modules.append(nn.ReLU())
        in_ch = out_ch

    out_ch = out_dims
    modules.append(nn.Linear(in_ch, out_ch))
    return nn.Sequential(*modules)

def build_mlp(cfg, in_dims, out_dims):
    # print(cfg)
    if cfg.use_tcnn and cfg.n_hidden_layers > 0:
        return build_tcnn_network(cfg, in_dims, out_dims)
    else:
        return build_nn_network(cfg, in_dims, out_dims)
