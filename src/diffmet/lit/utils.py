from typing import Type

# FIXME importlib?
def get_class(class_path: str) -> Type:
    """
    adapted from https://github.com/Lightning-AI/pytorch-lightning/blob/2.2.1/src/lightning/pytorch/cli.py#L730-L747
    """
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    return getattr(module, class_name)
