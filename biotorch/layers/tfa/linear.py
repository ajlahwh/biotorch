import torch
import torch.nn as nn
import biotorch.layers.tfa_constructor as tfa_constructor



class Linear(tfa_constructor.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        if layer_config is None:
            layer_config = {}
        layer_config["type"] = "tfa"

        super(Linear, self).__init__(in_features, out_features, bias, layer_config)
