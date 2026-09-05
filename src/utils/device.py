import torch


def get_device(device_name: str | None = None) -> torch.device:
    """
    Selects and returns an appropriate torch.device.

    Args:
        device_name: Optional explicit device name ('cuda', 'mps', 'cpu', 'cuda:0', etc.).
                     If None or 'auto', automatically selects CUDA -> MPS -> CPU.

    Returns:
        torch.device: The selected PyTorch device.
    """
    if device_name and device_name.lower() != 'auto':
        device = torch.device(device_name)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        return device

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
