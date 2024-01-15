import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import requests
import torch
from torch import nn

def load_model(model_name: str,
               model_dir: Union[str, Path] = './models',
               dataset: Union[str,
                              BenchmarkDataset] = BenchmarkDataset.cifar_10,
               threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
               norm: Optional[str] = None) -> nn.Module:
    """Loads a model from the model_zoo.

     The model is trained on the given ``dataset``, for the given ``threat_model``.

    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it
      overrides ``threat_model``

    :return: A ready-to-used trained model.
    """

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    if norm is None:
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_ = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
    model_path = model_dir_ / f'{model_name}.pt'

    models = all_models[dataset_][threat_model_]

    if not isinstance(models[model_name]['gdrive_id'], list):
        model = models[model_name]['model']()
        if dataset_ == BenchmarkDataset.imagenet and 'Standard' in model_name:
            return model.eval()
        
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        if not os.path.isfile(model_path):
            download_gdrive(models[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if 'Kireev2021Effectiveness' in model_name or model_name == 'Andriushchenko2020Understanding':
            checkpoint = checkpoint['last']  # we take the last model (choices: 'last', 'best')
        try:
            # needed for the model of `Carmon2019Unlabeled`
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'],
                                                   'module.')
            # needed for the model of `Chen2020Efficient`
            state_dict = rm_substr_from_state_dict(state_dict,
                                                   'model.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')

        if dataset_ == BenchmarkDataset.imagenet:
            # so far all models need input normalization, which is added as extra layer
            state_dict = add_substr_to_state_dict(state_dict, 'model.')
        
        model = _safe_load_state_dict(model, model_name, state_dict, dataset_)

        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model = models[model_name]['model']()
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        for i, gid in enumerate(models[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i),
                                    map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(
                    checkpoint['state_dict'], 'module.')
            except KeyError:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

            model.models[i] = _safe_load_state_dict(model.models[i],
                                                    model_name, state_dict,
                                                    dataset_)
            model.models[i].eval()

        return model.eval()
