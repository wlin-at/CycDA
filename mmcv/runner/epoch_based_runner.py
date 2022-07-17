# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            ####################################################################################################################################################
            ########################################################################## train one iteration
            ####################################################################################################################################################
            kwargs.update( {'iter': self._iter,
                            'max_iters' : self._max_iters})
            # print(f'iter_now {self._iter} max_iters {self._max_iters}')

            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)    ########  got to mmaction/models/recognizers/base.py     def train_step
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        ####################################################################################################################################################
        ########################################################################## train one epoch
        ####################################################################################################################################################
        # todo in the case of DA,   kwargs contains the DA_config
        # in the case of DA,   data_loader for training is tuple of source train and target train dataloader
        self.model.train()
        self.mode = 'train'

        if 'DA_config' in kwargs:
            self.DA_config = kwargs['DA_config']
            # todo in the case of DA,  there are source train dataloader and target train dataloader
            DA_config = kwargs['DA_config']

            #  prepare dataloaders
            dataload_iter = DA_config.dataload_iter
            w_pseudo = DA_config.w_pseudo

            source_train_loader = data_loader
            target_train_loader = self.target_train_loader
            dataloader_list = [source_train_loader, target_train_loader]



            experiment_type = DA_config.experiment_type
            if DA_config.model_type == 'RecognizerI3dDA':
                weight_clip_clspred = DA_config.weight_clip_clspred
                if_use_source_batch = not (experiment_type == 'DA' and weight_clip_clspred == 0)
            elif DA_config.model_type =='RecognizerTempAgg':
                weight_clip_clspred, weight_vid_clspred, weight_clip_domainpred, weight_vid_domainpred = DA_config.weight_clip_clspred, DA_config.weight_vid_clspred, DA_config.weight_clip_domainpred, DA_config.weight_vid_domainpred
                if_use_source_batch = not ( experiment_type == 'DA' and weight_clip_clspred == 0 and weight_vid_clspred == 0 and weight_clip_domainpred == 0 and weight_vid_domainpred == 0)
            if_use_target_batch = not experiment_type in ['source_only', 'target_only']

            if if_use_source_batch and not if_use_target_batch: # only source batch will be used
                dataload_main_idx = 0
            elif if_use_target_batch and not if_use_source_batch: # only target batch will be used
                dataload_main_idx = 1
            elif if_use_source_batch and if_use_target_batch:
                # todo if both source batch and target batch are used,  we decide the main dataloader based on length of dataloader and  dataload_iter
                max_dataload_idx = 0 if len(source_train_loader) >= len(target_train_loader) else 1
                if dataload_iter == 'min':
                    dataload_main_idx = 1- max_dataload_idx
                elif dataload_iter == 'max':
                    dataload_main_idx = max_dataload_idx
                # todo initialize dataloader_sec
                dataloader_sec = dataloader_list[1 - dataload_main_idx]  # secondary dataloader,
                dataloader_sec_iterator = iter( dataloader_sec)  # in the beginning of each epoch, reset the secondary dataloader


            dataloader_main = dataloader_list[dataload_main_idx] # main dataloader, determines the number of iterations in training
            self.data_loader = dataloader_main

            self._max_iters = self._max_epochs * len( dataloader_main )
            self.call_hook('before_train_epoch')
            time.sleep(2)  # Prevent possible deadlock during epoch transition

            for i, data_dict_main in enumerate( dataloader_main):  # enumerate() automatically generates a new iterator
                if if_use_source_batch and if_use_target_batch:
                    # todo if both source batch and target batch are used, we load data from dataloader_sec
                    if i == 0:
                        print(f'In the epoch beginning. The secondary dataloader of length {len(dataloader_sec)} is reset for a new iterator!')
                        # in the beginning of each epoch, reset the secondary dataloader
                        dataloader_sec_iterator = iter(dataloader_sec)
                    try:
                        data_dict_sec = next(dataloader_sec_iterator)
                    except StopIteration:
                        # when the secondary dataloader runs out of samples, reset the iterator for this dataloader
                        print(f'The secondary dataloader of length {len(dataloader_sec)} is reset for a new iterator!')
                        dataloader_sec_iterator = iter(dataloader_sec)
                        data_dict_sec = next(dataloader_sec_iterator)
                else:
                    # todo  if only source batch is used or only target batch is used, WE DO NOT LOAD DATA FROM dataloader_sec  at all   to SPEED UP
                    data_dict_sec = None
                if dataload_main_idx == 0:  #  main dataloader is source train dataloader
                    data_dict_source, data_dict_target = data_dict_main, data_dict_sec
                elif dataload_main_idx == 1:
                    data_dict_source, data_dict_target = data_dict_sec, data_dict_main

                self._inner_iter = i
                self.call_hook('before_train_iter' )
                ####################################################################################################################################################
                ########################################################################## train one iteration
                ####################################################################################################################################################
                self.run_iter( (data_dict_source, data_dict_target), train_mode= True, **kwargs )
                self.call_hook('after_train_iter')
                self._iter += 1


        else:
            self.data_loader = data_loader
            self._max_iters = self._max_epochs * len(self.data_loader)
            self.call_hook('before_train_epoch')
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for i, data_batch in enumerate(self.data_loader):
                # data_batch is a dictory of  'imags' and 'label'
                self._inner_iter = i
                self.call_hook('before_train_iter')
                ####################################################################################################################################################
                ########################################################################## train one iteration
                ####################################################################################################################################################
                # todo in the case of DA,   kwargs contains the DA_config
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                self._iter += 1

        self.call_hook('after_train_epoch')   #  after each training epoch, check if do validation
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.

            # todo in the case of DA,   kwargs contains the DA_config
        """
        assert isinstance(data_loaders, list)
        if 'DA_config' in kwargs:
            self.DA_config = kwargs['DA_config']
            self.cfg_target_train = kwargs['cfg_target_train']
            self.cfg_target_train_for_inference = kwargs['cfg_target_train_for_inference']
            # self.cfg_val = kwargs['cfg_val']
            self.train_dataloader_setting = kwargs['train_dataloader_setting']
            self.val_dataloader_setting = kwargs['val_dataloader_setting']

        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')


        ####################################################################################################################################################
        ########################################################################## train for all epochs
        ####################################################################################################################################################
        # criterion
        if 'DA_config' in kwargs:
            kwargs['DA_config'].update(
                {'ce': torch.nn.CrossEntropyLoss().cuda(),
                 'ce_d': torch.nn.CrossEntropyLoss().cuda(),
                 'ce_wo_avg': torch.nn.CrossEntropyLoss(reduction='none') # no reduction applied, manual averaging required. The element-wise weights can be added to the loss
                 }
            )

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                #  workflow is  (train, 1)
                mode, epochs = flow  #  epochs = 1
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):  # epochs= 1 by default
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break

                    ####################################################################################################################################################
                    ########################################################################## train one epoch
                    ####################################################################################################################################################
                    # todo in the case of DA,   kwargs contains the DA_config
                    epoch_runner(data_loaders[i], **kwargs)   ####  go to  def train  in line 41

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        filename_tmpl = 'epoch_{}'+f'_{self.timestamp}'+'.pth'
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
