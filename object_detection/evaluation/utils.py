import pathlib
import numpy as np
import torch
from typing import List
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def get_filenames_of_path(path: List[pathlib.Path], ext: str = '*'):
    """
    Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

def color_mapping_func(labels, mapping):
    """Maps an label (integer or string) to a color"""
    color_list = [mapping[value] for value in labels]
    return color_list


def enable_gui_qt():
    """Performs the magic command %gui qt"""
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.magic('gui qt')

def from_file_to_BoundingBox(file_name: pathlib.Path, groundtruth: bool = True):
    """Returns a list of BoundingBox objects from groundtruth or prediction."""
    from metrics.bounding_box import BoundingBox
    from metrics.enumerators import BBFormat, BBType

    file = torch.load(file_name)
    labels = file['labels']
    boxes = file['boxes']
    scores = file['scores'] if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [BoundingBox(image_name=file_name.stem,
                        class_id=l,
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


def from_dict_to_BoundingBox(file: dict, name: str, groundtruth: bool = True):
    """Returns list of BoundingBox objects from groundtruth or prediction."""
    from metrics.bounding_box import BoundingBox
    from metrics.enumerators import BBFormat, BBType

    labels = file['labels']
    boxes = file['boxes']
    scores = np.array(file['scores'].cpu()) if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [BoundingBox(image_name=name,
                        class_id=int(l),
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


def log_packages_neptune(neptune_logger):
    """Uses the neptunecontrib.api to log the packages of the current python env."""
    from neptunecontrib.api import log_table
    import pandas as pd

    import importlib_metadata

    dists = importlib_metadata.distributions()
    packages = {idx: (dist.metadata['Name'], dist.version) for idx, dist in enumerate(dists)}

    packages_df = pd.DataFrame.from_dict(packages, orient='index', columns=['package', 'version'])

    log_table(name='packages', table=packages_df, experiment=neptune_logger.experiment)


def log_mapping_neptune(mapping: dict, neptune_logger):
    """Uses the neptunecontrib.api to log a class mapping."""
    from neptunecontrib.api import log_table
    import pandas as pd

    mapping_df = pd.DataFrame.from_dict(mapping, orient='index', columns=['class_value'])
    log_table(name='mapping', table=mapping_df, experiment=neptune_logger.experiment)


def log_model_neptune(checkpoint_path: pathlib.Path,
                      save_directory: pathlib.Path,
                      name: str,
                      neptune_logger):
    """Saves the model to disk, uploads it to neptune and removes it again."""
    import os
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['hyper_parameters']['model']
    torch.save(model.state_dict(), save_directory / name)
    neptune_logger.experiment.set_property('checkpoint_name', checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(save_directory / name))
    if os.path.isfile(save_directory / name):
        os.remove(save_directory / name)


def log_checkpoint_neptune(checkpoint_path: pathlib.Path, neptune_logger):
    neptune_logger.experiment.set_property('checkpoint_name', checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(checkpoint_path))
