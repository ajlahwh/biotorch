import biotorch.models.ffn_net as models
from biotorch.models.utils import create_ffn_biomodel


MODE = 'backpropagation'
MODE_STRING = 'Backpropagation'


def ffnnet(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config: dict = None):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting FFN_net to {} mode'.format(MODE_STRING))
    return create_ffn_biomodel(models.ffn_mnist, MODE, layer_config, pretrained, progress, num_classes)