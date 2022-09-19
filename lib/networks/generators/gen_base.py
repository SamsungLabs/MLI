__all__ = ['GeneratorBase']

import copy
import logging
from abc import abstractmethod

from torch import nn

from lib.networks.generators import gen_parts
from lib.utils.base import get_total_data_dim

logger = logging.getLogger(__name__)


class GeneratorBase(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._init_modules(**self._build_modules(self.params['modules']))

    @staticmethod
    def _build_modules(params):
        modules = {}
        for module_name, module_config in params.items():
            module_config_copy = copy.deepcopy(module_config)
            architecture = module_config_copy.pop('architecture')
            frozen = module_config_copy.pop('frozen', False)
            module_config_copy.pop('discard_pretrain', False)

            if 'input_data' in module_config_copy:
                module_config_copy['input_dim'] = get_total_data_dim(module_config_copy['input_data'])
                module_config_copy.pop('input_data')
            if 'output_data' in module_config_copy:
                module_config_copy['output_dim'] = get_total_data_dim(module_config_copy['output_data'])
                module_config_copy.pop('output_data')

            logger.debug(f'Building {module_name} with {architecture}')
            modules[module_name] = getattr(gen_parts, architecture)(**module_config_copy)

            if frozen:
                for param in modules[module_name].parameters():
                    param.requires_grad = False
                logger.debug(f'{module_name} was frozen')

        return modules

    @abstractmethod
    def _init_modules(self, *args, **kwargs):
        """
        Init generator modules
        """
        raise NotImplementedError
