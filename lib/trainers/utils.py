import os
import re
from typing import Union, Optional

import torch

from lib import trainers
from lib.utils.base import get_latest_model_name


def create_trainer_load_weights(config: dict,
                                checkpoints_dir: str,
                                iteration: Optional[int] = None,
                                use_ema: bool = False,
                                device: Union[str, torch.device] = 'cuda',
                                ):
    loaded_iteration = iteration
    trainer = getattr(trainers, config['trainer'])(config, eval_mode=True, device=device)

    state_dicts = []
    for model_part in 'gen':
        model_part_weight_path = os.path.join(checkpoints_dir, f'model.{model_part}_{iteration:08d}.pt')
        current_state_dict = torch.load(model_part_weight_path, map_location=device)
        current_state_dict = {f'{model_part}.' + re.sub(r'^module\.', '', k): v for k, v in current_state_dict.items()}
        state_dicts.append(current_state_dict)

    full_state_dict = {}
    for current_state_dict in state_dicts:
        full_state_dict.update(current_state_dict)

    trainer.load_state_dict(full_state_dict, strict=False)
    trainer.ema_inference = use_ema

    return trainer, loaded_iteration


def create_trainer_load_weights_from_config(config: dict,
                                            checkpoints_dir: str,
                                            iteration: Optional[int] = None,
                                            use_ema: bool = False,
                                            device: Union[str, torch.device] = 'cuda',
                                            ):
    loaded_iteration = iteration
    trainer = getattr(trainers, config['trainer'])(config, eval_mode=True, device=device)

    state_dicts = []
    for model_part in config['models']:
        if model_part == 'dis':
            continue
        if iteration is None:
            model_part_weight_path = get_latest_model_name(checkpoints_dir, model_part)
            loaded_iteration = int(model_part_weight_path.split('_')[-1][:-3])
        else:
            model_part_weight_path = os.path.join(checkpoints_dir, f'model.{model_part}_{iteration:08d}.pt')
        current_state_dict = torch.load(model_part_weight_path, map_location=device)

        for module_name in config['models'][model_part]['modules'].keys():
            if config['models'][model_part]['modules'][module_name].get('discard_pretrain', False):
                keys = list(current_state_dict.keys())
                for key in keys:
                    if module_name in key:
                        del (current_state_dict[key])

        current_state_dict = {f'{model_part}.' + re.sub(r'^module\.', '', k): v for k, v in current_state_dict.items()}
        state_dicts.append(current_state_dict)

    if use_ema:
        for model_name in getattr(trainer, 'ema_models_list', []):
            current_state_dict = torch.load(os.path.join(checkpoints_dir,
                                                         f'model_ema.{model_name}.pt'),
                                            map_location=device)
            current_state_dict = {f'{model_name}.' + re.sub(r'^module\.', '', k): v
                                  for k, v in current_state_dict.items()}
            state_dicts.append(current_state_dict)

    full_state_dict = {}
    for current_state_dict in state_dicts:
        full_state_dict.update(current_state_dict)

    # TODO implement inference load state method in lib.trainers.trainer_base.TrainerBase
    trainer.load_state_dict(full_state_dict, strict=False)
    trainer.ema_inference = use_ema

    return trainer, loaded_iteration
