import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.optim
try:
    from apex.parallel import DistributedDataParallel
    from apex import amp
except ImportError:
    pass
try:
    import horovod.torch as hvd
except ImportError:
    pass

from lib.utils.base import get_latest_model_name, clone_module
from lib import networks
from lib import modules
from lib.networks.generators import gen_parts

logger = logging.getLogger(__name__)


class TrainerBase(nn.Module):
    def __init__(self, params, eval_mode=False, device='cuda'):
        super().__init__()
        self.params = params
        self.gradient_update_period = params.get('gradient_update_iter', 1)
        self.models_dict = defaultdict(list)
        self.datasets_dict = {}
        self.dataloaders_dict = {}
        self.device = device
        self.eval_mode = eval_mode

        models_named_parameters = self._init_models()
        self._init_auxiliary_modules(eval_mode=eval_mode)

        self.ema_models_list = self._init_avg_models()
        self.ema_inference = params.get('ema_inference', False)

        if not eval_mode:
            self._init_optimizers(models_named_parameters)
            self._init_schedulers()
        else:
            self.eval()

    def _init_auxiliary_modules(self, eval_mode=False):
        for model_name, model_config in self.params.get('auxiliary_modules', {}).items():
            architecture = model_config.pop('architecture')
            disable_in_eval_mode = model_config.get('disable_in_eval_mode', False)

            if disable_in_eval_mode and eval_mode:
                logger.debug(f'Disable {model_name} with: {architecture} during eval mode')
                continue

            model_class = getattr(modules, architecture, None)
            if model_class is None:
                model_class = getattr(gen_parts, architecture, None)
            if model_class is None:
                raise AttributeError(f'Cannot find architecture called {architecture}')
            logger.debug(f'Building {model_name} with: {architecture}')
            setattr(self,
                    model_name,
                    model_class(**model_config).to(self.device)
                    )

    def _init_models(self):
        named_parameters = defaultdict(list)
        for model_name, model_config in self.params['models'].items():
            architecture = self.params['models'][model_name]['architecture']
            logger.debug(f'Building {model_name} with: {architecture}')
            setattr(self,
                    model_name,
                    getattr(networks, architecture)(model_config).to(self.device)
                    )
            named_parameters[model_config['optimizer_group']].extend(getattr(self, model_name).named_parameters())
            self.models_dict[model_config['optimizer_group']].append(model_name)
        return named_parameters

    def _init_optimizers(self, models_named_parameters):
        use_apex = self.params.get('use_apex', True)
        use_horovod = self.params.get('use_horovod', False)
        assert not (use_horovod and use_apex)

        self.optimizers = {}
        for group, config in self.params['optimizers'].items():
            opt_type = config.pop('type')

            if use_horovod:
                lr_scaler = 1
                use_adasum = config.pop('use_adasum', False)
                if not use_adasum:
                    lr_scaler = hvd.size()
                elif hvd.nccl_built():
                    lr_scaler = hvd.local_size()
                logger.info(f'Scaling optimizer lr for horovod by {lr_scaler}')
                config['lr'] *= lr_scaler
            elif use_apex:
                lr_scaler = int(os.environ.get('WORLD_SIZE', '1'))
                logger.info(f'Scaling optimizer lr for apex by {lr_scaler}')
                config['lr'] *= lr_scaler

            models_parameters = [par for name, par in models_named_parameters[group]]
            optimizer = getattr(torch.optim, opt_type)(models_parameters, **config)

            if use_apex:
                amped_models, self.optimizers[group] = amp.initialize(
                    [getattr(self, cur_model) for cur_model in self.models_dict[group]], optimizer,
                    opt_level=self.params['opt_level'], max_loss_scale=1024)
                for cur_model_name, cur_amped_model in zip(self.models_dict[group], amped_models):
                    setattr(self, cur_model_name, DistributedDataParallel(cur_amped_model, delay_allreduce=True))
                self.optimizers[group].custom_lr_scaler = lr_scaler

            elif use_horovod:
                for cur_model in self.models_dict[group]:
                    hvd.broadcast_parameters(getattr(self, cur_model).state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)
                self.optimizers[group] = hvd.DistributedOptimizer(optimizer,
                                                                  named_parameters=models_named_parameters[group],
                                                                  compression=hvd.Compression.none,
                                                                  op=hvd.Adasum if use_adasum else hvd.Average,
                                                                  gradient_predivide_factor=1.,
                                                                  )

                self.optimizers[group].use_adasum = use_adasum
                self.optimizers[group].custom_lr_scaler = lr_scaler
            else:
                self.optimizers[group] = optimizer

    def _init_schedulers(self, iterations=-1):
        use_horovod = self.params.get('use_horovod', False)
        use_apex = self.params.get('use_apex', True)
        self.schedulers = {}
        for group, config in self.params['schedulers'].items():
            sch_type = config.pop('type')
            for option_name in config:
                if (use_horovod or use_apex) and option_name.endswith('_lr'):
                    # Warning: This tuning w.r.t. the number of GPUs was implemented
                    # for torch.optim.lr_scheduler.CyclicLR and may not have the expected effect for other schedulers
                    logger.info(f'Scaling scheduler parameter {option_name} for horovod/apex '
                                f'by {self.optimizers[group].custom_lr_scaler}')
                    config[option_name] *= self.optimizers[group].custom_lr_scaler
            self.schedulers[group] = getattr(torch.optim.lr_scheduler, sch_type)(self.optimizers[group],
                                                                                 last_epoch=iterations,
                                                                                 **config)
            config['type'] = sch_type

    def _init_losses(self):
        self.losses = defaultdict(dict)
        self.gradient_info = {}

    def _unpack_data(self, data, mode):
        """

        :param data: data from dataset
        :param mode: unpack mode eval/train
        :return:
        """
        raise NotImplementedError

    def backward(self, parameter_group: str):
        grad_norm_thres: Optional[float] = self.params.get('clip_grad_norm')
        grad_value_thres: Optional[float] = self.params.get('clip_grad_value')
        assert (grad_norm_thres is None) or (grad_value_thres is None)
        use_apex = self.params.get('use_apex', True)
        use_horovod = self.params.get('use_horovod', False)

        if use_apex:
            with amp.scale_loss(self.losses[parameter_group]['total'],
                                self.optimizers[parameter_group]) as scaled_loss:
                scaled_loss.backward()
            parameters = amp.master_params(self.optimizers[parameter_group])
        else:
            self.losses[parameter_group]['total'].backward()
            if use_horovod:
                self.optimizers[parameter_group].synchronize()
            parameters = self.optimizers[parameter_group].param_groups[0]['params']

        if grad_norm_thres is not None:
            clip_grad_norm_(parameters, grad_norm_thres)
        elif grad_value_thres is not None:
            clip_grad_value_(parameters, grad_value_thres)

        grads = []
        for param in parameters:
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
        if grads:
            grads = torch.cat(grads)
            grad_norm, grad_max_val = grads.norm(), grads.abs().max().item()
            del grads
            torch.cuda.empty_cache()
            self.gradient_info[f'grad_norm/optimizer_group/{parameter_group}'] = grad_norm
            self.gradient_info[f'grad_max_val/optimizer_group/{parameter_group}'] = grad_max_val

    def optimizer_step(self, parameter_group: str):
        use_horovod = self.params.get('use_horovod', False)
        optimizer = self.optimizers[parameter_group]
        if use_horovod and not getattr(optimizer, 'use_adasum', False):
            with optimizer.skip_synchronize():
                optimizer.step()
        else:
            optimizer.step()

    def save_gradient_norms_module(self, module_name: str) -> None:
        use_apex = self.params.get('use_apex', False)
        module = getattr(self, module_name)
        if use_apex:
            module = module.module
        for child_name, child_module in module.named_children():
            grads = []
            for param in child_module.parameters():
                if param.grad is not None:
                    grads.append(param.grad.data.view(-1))
            if grads:
                grads = torch.cat(grads)
                grad_norm, grad_max_val = grads.norm(), grads.abs().max().item()
                del grads
                torch.cuda.empty_cache()
                self.gradient_info[f'grad_norm/module/{module_name}/{child_name}'] = grad_norm
                self.gradient_info[f'grad_max_val/module/{module_name}/{child_name}'] = grad_max_val

    def update(self, data, iteration) -> Optional[str]:
        self._init_losses()

        if self.gradient_update_period == 1 or iteration % self.gradient_update_period == 1 or iteration == 0:
            for opt_group in self.optimizers.values():
                opt_group.zero_grad()

        return self.update_step(iteration, *self._unpack_data(data, 'train'))

    def update_step(self, *args):
        """
        Compute losses and backward.
        :param data:
        :return:
        """
        raise NotImplementedError

    def update_learning_rate(self):
        """
        Update learning rates with schedulers.
        :return:
        """
        for current_scheduler in self.schedulers.values():
            current_scheduler.step()

    def _aggregate_losses(self, optimiser_group, iteration=None) -> bool:
        """
        Compute final loss by aggregating losses with weights.
        :param optimiser_group:
        :return:
        """
        self.losses[optimiser_group]['total'] = 0
        warm_up_params = self.params.get('losses_warm_up', None)
        for loss_name, weight in self.params['weights'][optimiser_group].items():
            curr_loss = weight * self.losses[optimiser_group].get(loss_name, 0)
            if iteration is not None and warm_up_params is not None:
                if optimiser_group in warm_up_params:
                    if warm_up_params[optimiser_group].get(loss_name, 0) > iteration:
                        curr_loss = curr_loss * 0
            self.losses[optimiser_group]['total'] += curr_loss

        return torch.isnan(self.losses[optimiser_group]['total']).item()

    def _compute_metrics(self, output, data):
        """
        Compute metrics for batch.
        :param output:
        :param data:
        :return: dict
        """
        raise NotImplementedError

    def _aggregate_metrics(self, metrics):
        """
        Aggregate metrics for several batches.
        :param metrics: list of dicts
        :return: dict
        """
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, dataloader, num_repeat=1):
        """
        Compute metrics for validation axis.
        :param dataloader: torch.utils.axis.Dataloader
        :param num_epochs: int number of epochs to compute metric
        :return: dict
        """
        metrics = []
        training_status = self.training
        self.eval()
        for _ in range(num_repeat):
            for data in dataloader:
                result = self.forward(*self._unpack_data(data, mode='eval'))
                metrics.append(self._compute_metrics(result, data))
                del result
        self.train(training_status)
        torch.cuda.empty_cache()
        return self._aggregate_metrics(metrics)

    def _compute_visualisation(self, output, data):
        """
        Compute visualisation for batch.
        :param output:
        :param data:
        :return: list of PIL.Images
        """
        raise NotImplementedError

    def _aggregate_visualisation(self, visualisations):
        """
        Aggregate visualisation for several batches.
        :param visualisations: list of PIL.Images
        :return: PIL.Image
        """
        raise NotImplementedError

    @torch.no_grad()
    def visualise(self, dataloader):
        """
        Generate image with results for visualise axis.
        :param dataloader: torch.utils.axis.Dataloader
        :return: PIL.Image
        """
        visualisations = []
        training_status = self.training
        self.eval()
        for data in dataloader:
            result = self.forward(*self._unpack_data(data, 'eval'))
            visualisations.extend(self._compute_visualisation(result, data))
            del result
        self.train(training_status)
        torch.cuda.empty_cache()
        return self._aggregate_visualisation(visualisations)

    def _init_avg_models(self) -> List[str]:
        ema_models = ['gen_ema']

        self.gen_ema = clone_module(self.gen)
        return ema_models

    def resume(self, checkpoint_dir: str, iterations: int = None):
        # Load generators
        models = self.params['models']
        discard_something = False
        for model_name in self.params['models']:
            latest_name = get_latest_model_name(checkpoint_dir, model_name)
            if latest_name is None:
                logger.info('Unable to load checkpoints: cannot find them. Starting training from scratch...')
                return -1

            if iterations is None:
                iterations = int(latest_name.split('_')[-1][:-3])
                model_path = latest_name
            else:
                model_path = f"{latest_name[:-11]}{iterations:08d}.pt"  # 11 = 8 (iter num) + 3 (.pt)
            logger.info(model_path)
            state_dict = torch.load(model_path,
                                    map_location=self.device,
                                    )
            for module_name in self.params['models'][model_name]['modules'].keys():
                if self.params['models'][model_name]['modules'][module_name].get('discard_pretrain', False):
                    keys = list(state_dict.keys())
                    discard_something = True
                    for key in keys:
                        if module_name in key:
                            del(state_dict[key])

            #  To overcome apex issue
            if not self.params.get('use_apex', False):
                state_dict = {f'{model_name}.' + re.sub(r'^module\.', '', k): v for k, v in state_dict.items()}
            else:
                state_dict = {k.replace(f'{model_name}', 'module'): v for k, v in state_dict.items()}
            logger.info(f'Load {model_name}')
            getattr(self, model_name).load_state_dict(state_dict, strict=False)

        if not self.eval_mode and not discard_something:
            for name, optimizer in self.optimizers.items():
                optimizer.load_state_dict(
                    torch.load(os.path.join(checkpoint_dir, f'optimizer.{name}_{iterations:08d}.pt'),
                               map_location=self.device,
                               )
                    )

            # Reinitialize schedulers
            self._init_schedulers(iterations=iterations)
            logger.info(f'Resume from iteration {iterations:d}')

        if self.params.get('use_apex', True):
            apex_checkpoint = torch.load(os.path.join(checkpoint_dir, f'amp.{iterations:08d}.pt'),)
            amp.load_state_dict(apex_checkpoint)
            logger.info(f'Load apex')

        if iterations > -1 and not discard_something:
            for model_name in self.ema_models_list:
                if not os.path.exists(os.path.join(checkpoint_dir, f'model_ema.{model_name}.pt')):
                    continue

                state_dict = torch.load(
                    os.path.join(checkpoint_dir, f'model_ema.{model_name}.pt'),
                    map_location=self.device,
                )

                getattr(self, model_name).load_state_dict(state_dict, strict=False)

        return iterations

    def save(self, snapshot_dir, iterations):
        for model_name in self.params['models']:
            torch.save(getattr(self, model_name).state_dict(),
                       os.path.join(snapshot_dir, f'model.{model_name}_{iterations:08d}.pt')
                       )
        for name, optimizer in self.optimizers.items():
            torch.save(optimizer.state_dict(),
                       os.path.join(snapshot_dir, f'optimizer.{name}_{iterations:08d}.pt')
                       )

        if self.params.get('use_apex', True):
            apex_checkpoint = amp.state_dict()
            torch.save(apex_checkpoint, os.path.join(snapshot_dir, f'amp.{iterations:08d}.pt'))

        for model_name in self.ema_models_list:
            torch.save(getattr(self, model_name).state_dict(),
                       os.path.join(snapshot_dir, f'model_ema.{model_name}.pt')
                       )

    @torch.no_grad()
    def inference(self, data):
        pass

    @staticmethod
    def _perceptual_preprocessing(batch, max_value: float = 255):
        """
        The scaling procedure for all the pretrained models from torchvision is described in the docs
        https://pytorch.org/docs/stable/torchvision/models.html
        """
        batch = torch.clamp(batch[:, :3], 0, max_value) / max_value  # [0, 255] -> [0, 1]
        mean = torch.tensor([.485, .456, .406], dtype=batch.dtype, device=batch.device)[None, :, None, None]
        batch = batch.sub(mean)  # subtract mean
        std = torch.tensor([.229, .224, .225], dtype=batch.dtype, device=batch.device)[None, :, None, None]
        batch = batch.div(std)
        return batch

    @staticmethod
    def average_models(model1, model2, alpha_decay=0.9999):
        """
        To use this method at generators:
                self.average_models(self.gen_ema, your_generator, alpha_decay=self.params.get('decay_rate', 0.99))
        """
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1:
            par1[k].data.mul_(alpha_decay).add_(1 - alpha_decay, par2[k].data)
