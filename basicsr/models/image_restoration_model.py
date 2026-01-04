import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import os
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
import math
import numpy as np


class ClassificationModel(BaseModel):
    def __init__(self, opt):
        super(ClassificationModel, self).__init__(opt)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.net_g.to(self.device)
        if opt.get('compile', False):
            self.net_g = torch.compile(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                self.net_g, load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params')
            )

        if self.is_train:
            self.init_training_settings()

        self.best_acc = 0.0
        self.acc_value = 0.0

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.cri_ce = torch.nn.CrossEntropyLoss().to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                pass

        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g'].pop('lr')
        train_opt['optim_g']['lr'] = lr

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params, 'lr': lr}],
                                                **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params, 'lr': lr}],
                                                 **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params, lr=lr,
                                               **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.img = data['img'].to(self.device)
        if 'label' in data:
            self.label = data['label'].to(self.device)

    def optimize_parameters(self, current_iter, tb_logger=None):
        self.optimizer_g.zero_grad()

        preds = self.net_g(self.img)

        l_ce = self.cri_ce(preds, self.label)

        l_ce.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', False)
        if use_grad_clip:
            clip_value = self.opt['train'].get('grad_clip_value', 0.01)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), clip_value)

        self.optimizer_g.step()

        with torch.no_grad():
            _, predicted = torch.max(preds, 1)
            correct = (predicted == self.label).sum().item()
            total = self.label.size(0)
            acc = torch.tensor(100.0 * correct / total if total > 0 else 0.0, dtype=l_ce.dtype, device=l_ce.device)

        loss_dict = OrderedDict()
        loss_dict['l_ce'] = l_ce
        loss_dict['acc'] = acc

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.img)
        self.net_g.train()
        return self.output

    def validation(self, dataloader, current_iter, tb_logger, save_img=False, rgb2bgr=False, use_image=False):
        self.net_g.eval()

        total_correct = 0
        total_samples = 0
        val_loss = 0.0

        with torch.no_grad():
            for val_data in dataloader:
                self.feed_data(val_data, is_val=True)
                outputs = self.net_g(self.img)

                loss = self.cri_ce(outputs, self.label)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == self.label).sum().item()
                total_samples += self.label.size(0)


        val_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
        avg_val_loss = val_loss / len(dataloader) if len(dataloader) > 0 else float('inf')

        self.acc_value = val_acc

        logger = get_root_logger()
        logger.info(f'Validation #{current_iter//self.opt["val"]["val_freq"] if self.opt.get("val") else current_iter}: Iter {current_iter}, Acc: {val_acc:.2f}%, Loss: {avg_val_loss:.4f}')

        if tb_logger:
            tb_logger.add_scalar('metrics/val_acc', val_acc, current_iter)
            tb_logger.add_scalar('metrics/val_loss', avg_val_loss, current_iter)

        self.net_g.train()
        return val_acc

    def save(self, epoch, current_iter, stage='', **kwargs):
        save_name = f'net_g_{stage}' if stage else 'net_g'
        self.save_network(self.net_g, save_name, current_iter)

        state = {
            'epoch': epoch,
            'iter': current_iter,
            'best_acc': self.best_acc,
            'optimizers': [],
            'schedulers': []
        }

        for optimizer in self.optimizers:
            state['optimizers'].append(optimizer.state_dict())
        for scheduler in self.schedulers:
            state['schedulers'].append(scheduler.state_dict())

        state.update(kwargs)

        save_filename = f'{current_iter}.state'
        save_path = os.path.join(self.opt['path']['training_states'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        self.best_acc = resume_state.get('best_acc', 0.0)

        super().resume_training(resume_state)

    def get_current_visuals(self):
        return OrderedDict()

    def get_current_log(self):
        return self.log_dict.copy() if self.log_dict else OrderedDict()

    def train(self):
        self.is_train = True
        if hasattr(self, 'net_g') and self.net_g is not None:
            self.net_g.train()
        else:
            raise AttributeError("The network (self.net_g) is not initialized properly.")

    def eval(self):
        self.is_train = False
        if hasattr(self, 'net_g') and self.net_g is not None:
            self.net_g.eval()
        else:
            raise AttributeError("The network (self.net_g) is not initialized properly.")
