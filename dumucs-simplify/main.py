"""

"""
import inspect
import warnings
from pathlib import Path

import yaml
import torch

from demucs import model_htdumucs

REMOTE_ROOT = Path(__file__).parent / 'remote'
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"


class ModelLoadingError(RuntimeError):
    pass


def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get('__quantized'):
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state['quantized'])
        else:
            # _check_diffq()
            from diffq import restore_quantized_state
            # This package implements different quantization strategies?
            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)
    return state


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, 'cpu')
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def get_model():
    """
    获取模型, 使用模型为 htdemucs_ft
    :return:
    """
    root: str = ''
    models_remote = models = {}
    remote_file_list = REMOTE_ROOT / 'files.txt'
    # 加载 models (虽然是固定的)
    for line in remote_file_list.read_text().split('\n'):
        line = line.strip()
        if line.startswith('#'):
            continue
        elif line.startswith('root:'):
            root = line.split(':', 1)[1].strip()
        else:
            sig = line.split('-', 1)[0]
            assert sig not in models
            models[sig] = ROOT_URL + root + line
    # 加载 model

    with open("remote/htdemucs_ft.yaml") as fp:
        bag = yaml.safe_load(fp)
    signatures = bag['models']
    models = []
    for sig in signatures:
        url = models_remote[sig]
        pkg = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)  # type: ignore
        m = load_model(pkg)
        models.append(m)
    weights = bag.get('weights')
    segment = bag.get('segment')

    model = model_htdumucs.BagOfModels(models, weights, segment)
    model.eval()

    return model


def 面(file_path: str):
    """

    :return:
    """
    model = get_model()


if __name__ == "__main__":
    面("")
