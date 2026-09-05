from src.models.autoencoder import Autoencoder
from src.models.autoencoder_old import AutoencoderOld
from src.models.autoencoder_attention import AutoencoderAttention
from src.models.bottle_net import BottleNetwork
from src.models.dense_net import DenseNet
from src.models.resnext import ResNext
from src.models.vgg import Vgg16, VggOutputs

_MODEL_REGISTRY = {
    'ae': Autoencoder,
    'autoencoder': Autoencoder,
    'ae_old': AutoencoderOld,
    'autoencoder_old': AutoencoderOld,
    'ae_attn': AutoencoderAttention,
    'autoencoder_attention': AutoencoderAttention,
    'attention': AutoencoderAttention,
    'bo': BottleNetwork,
    'bottlenet': BottleNetwork,
    'bottlenetwork': BottleNetwork,
    'dense': DenseNet,
    'densenet': DenseNet,
    'res': ResNext,
    'resnext': ResNext,
}


def get_model(model_type: str):
    """
    Factory function to instantiate a model by architecture name.

    Args:
        model_type (str): Short code or name ('ae', 'bo', 'res', 'dense', 'ae_attn', 'ae_old').

    Returns:
        torch.nn.Module: Instantiated network.
    """
    key = model_type.lower().strip()
    if key not in _MODEL_REGISTRY:
        supported = sorted(list(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model architecture '{model_type}'. Supported architectures: {supported}")
    return _MODEL_REGISTRY[key]()


def list_models():
    """Returns unique supported model names."""
    return sorted(list(set(_MODEL_REGISTRY.keys())))


__all__ = [
    'Autoencoder',
    'AutoencoderOld',
    'AutoencoderAttention',
    'BottleNetwork',
    'DenseNet',
    'ResNext',
    'Vgg16',
    'VggOutputs',
    'get_model',
    'list_models',
]
