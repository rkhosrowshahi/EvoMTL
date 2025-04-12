from collections import defaultdict
import pandas as pd
import torch, os, copy, torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import build_model_from_blocks_using_centers, build_model_from_blocks_using_gaussian, count_parameters, plot_pareto_front_and_population, ubp_cluster
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param,
                 save_path=None, load_path=None, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path
        self.weighting = weighting

        self.bilevel_methods = ['MOML', 'FORUM', 'AutoLambda']
        self.hybrid_methods = ['EVO']

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

        if self.weighting in self.bilevel_methods:
            self._prepare_tw(self.kwargs['weight_args']['outer_lr'])
        
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):

        weighting_class = weighting_method.__dict__['EW' if self.weighting in self.bilevel_methods + self.hybrid_methods else weighting] 
        architecture_class = architecture_method.__dict__[architecture]
        
        class MTLmodel(architecture_class, weighting_class):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)
        
    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def forward4loss(self, model, inputs, gts, return_preds=False):
        if not self.multi_input:
            preds = model(inputs)
            preds = self.process_preds(preds)
            losses = self._compute_loss(preds, gts)
        else:
            losses = torch.zeros(self.task_num).to(self.device)
            preds = {}
            for tn, task in enumerate(self.task_name):
                inputs_t, gts_t = inputs[task], gts[task]
                preds_t = model(inputs_t, task)
                preds_t = preds_t[task]
                preds_t = self.process_preds(preds_t, task)
                losses[tn] = self._compute_loss(preds_t, gts_t, task)
                if return_preds:
                    preds[task] = preds_t
        if return_preds:
            return losses, preds
        else:
            return losses

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False, **kwargs):
        if self.weighting in self.bilevel_methods:
            train_func = self.train_bilevel
        elif self.weighting in self.hybrid_methods:
            train_func = self.train_evo_using_centers
        else:
            train_func = self.train_singlelevel
        train_func(train_dataloaders, test_dataloaders, epochs, 
            val_dataloaders, return_weight, **kwargs)

    def train_singlelevel(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                train_losses = []
                for sample_num in range(3 if self.weighting in ['MoDo', 'SDMGrad'] else 1):
                    if not self.multi_input:
                        train_inputs, train_gts = self._process_data(train_loader)
                    else:
                        train_inputs, train_gts = {}, {}
                        for tn, task in enumerate(self.task_name):
                            train_input, train_gt = self._process_data(train_loader[task])
                            train_inputs[task], train_gts[task] = train_input, train_gt

                    train_losses_, train_preds = self.forward4loss(self.model, train_inputs, train_gts, return_preds=True)
                    train_losses.append(train_losses_)
                train_losses = torch.stack(train_losses).squeeze(0)

                if not self.multi_input:
                    self.meter.update(train_preds, train_gts)
                else:
                    for tn, task in enumerate(self.task_name):
                        self.meter.update(train_preds[task], train_gts[task], task)

                self.optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()

                if self.weighting == 'FAMO':
                    with torch.no_grad():
                        new_train_losses = self.forward4loss(self.model, train_inputs, train_gts, return_preds=False)
                        self.model.update_w(new_train_losses.detach())
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight


    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False, num_batch=None, verbose=True):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                total_loss = torch.zeros(self.task_num).to(self.device)
                total_batch = 0
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
                    total_loss += test_losses
                    total_batch += 1
                    if num_batch is not None and total_batch >= num_batch:
                        break
            else:
                total_loss = torch.zeros(self.task_num).to(self.device)
                for tn, task in enumerate(self.task_name):
                    total_batch = 0
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
                        total_loss[tn] += test_loss
                        total_batch += 1
                        if num_batch is not None and total_batch >= num_batch:
                            break
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode, verbose=verbose)
        improvement = self.meter.improvement
        losses = self.meter.get_losses()
        results = self.meter.get_results()
        self.meter.reinit()
        if return_improvement:
            return improvement
        return losses, results


    ### for bilevel methods
    def _prepare_tw(self, tw_lr):
        class TW(nn.Module):
            def __init__(self, task_num):
                super().__init__()
                self.weights = nn.Parameter(torch.FloatTensor(task_num))
                nn.init.constant_(self.weights, 1/task_num)
                
            def forward(self, loss):
                weight = F.softmax(self.weights, dim=-1)
                final_loss = torch.sum(torch.mul(weight, loss))
                return final_loss
        self.tw = TW(self.task_num).to(self.device)
        self.tw_optimizer = torch.optim.Adam(self.tw.parameters(), lr=tw_lr)

    def train_bilevel(self, train_dataloaders, test_dataloaders, epochs, 
                    val_dataloaders=None, return_weight=False):
        # we use different batch for inner loop and outer loop, thus we create a new train_dataloader with a half of batch size
        if self.multi_input:    
            new_train_dataloaders = {}
            for task in self.task_name:
                new_train_dataloaders[task] = torch.utils.data.DataLoader(
                                train_dataloaders[task].dataset,
                                batch_size=int(train_dataloaders[task].batch_size//2),
                                shuffle=True,
                                num_workers=train_dataloaders[task].num_workers,
                                drop_last=train_dataloaders[task].drop_last,
                                pin_memory=train_dataloaders[task].pin_memory,
                )
        else:
            try:
                new_train_dataloaders = torch.utils.data.DataLoader(
                                train_dataloaders.dataset,
                                batch_size=int(train_dataloaders.batch_size//2),
                                shuffle=True,
                                num_workers=train_dataloaders.num_workers,
                                drop_last=train_dataloaders.drop_last,
                                pin_memory=train_dataloaders.pin_memory,
                )
            except:
                # for QM9 only
                new_train_dataloaders = torch_geometric.loader.DataLoader(
                                train_dataloaders.dataset,
                                batch_size=int(train_dataloaders.batch_size//2),
                                shuffle=True,
                                num_workers=train_dataloaders.num_workers,
                                drop_last=train_dataloaders.drop_last,
                                pin_memory=train_dataloaders.pin_memory,
                )

        if self.weighting == 'FORUM':
            lambda_buffer = np.zeros([self.task_num])

        train_loader, train_batch = self._prepare_dataloaders(new_train_dataloaders)
        org_train_loader, _ = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        train_batch = int(train_batch / 2)

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                # get inner loop and outer loop data
                if not self.multi_input:
                    inner_x, inner_y = self._process_data(train_loader)
                    outer_x, outer_y = self._process_data(train_loader)
                else:
                    inner_x, inner_y = {}, {}
                    outer_x, outer_y = {}, {}
                    for tn, task in enumerate(self.task_name):
                        inner_x[task], inner_y[task] = self._process_data(train_loader[task])
                        outer_x[task], outer_y[task] = self._process_data(train_loader[task])

                if self.weighting == 'AutoLambda':
                    self.bacth_forward_AutoLambda(inner_x, inner_y, outer_x, outer_y, org_train_loader)
                elif self.weighting == 'MOML':
                    self.bacth_forward_MOML(inner_x, inner_y, outer_x, outer_y, org_train_loader)
                elif self.weighting == 'FORUM':
                    lambda_buffer = self.bacth_forward_FORUM(inner_x, inner_y, outer_x, outer_y, lambda_buffer, epoch)

            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight

    def bacth_forward_AutoLambda(self, inner_x, inner_y, outer_x, outer_y, org_train_dataloaders):
        r"""Auto-Lambda

        This method is proposed in `Auto-Lambda: Disentangling Dynamic Task Relationships (TMLR 2022) <https://openreview.net/forum?id=KKeCMim5VN>`_ and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/auto-lambda>`_.
        """
        assert self.kwargs['weight_args']['inner_step'] == 1, "AutoLambda is an approximated method for inner_step=1"

        def compute_hessian(self, grads, inner_x, inner_y):
            norm = torch.cat([g.view(-1) for g in grads]).norm()
            eps = 0.01 / norm

            # \theta+ = \theta + eps * d_model
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), grads):
                    p += eps * d

            losses = self.forward4loss(self.model, inner_x, inner_y)
            loss = self.tw(losses)
            d_weight_p = torch.autograd.grad(loss, self.tw.parameters())

            # \theta- = \theta - eps * d_model
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), grads):
                    p -= 2 * eps * d

            losses = self.forward4loss(self.model, inner_x, inner_y)
            loss = self.tw(losses)
            d_weight_n = torch.autograd.grad(loss, self.tw.parameters())

            # recover theta
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), grads):
                    p += eps * d

            hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
            return hessian

        try:
            inner_lr = self.scheduler.get_last_lr()[0]
        except:
            inner_lr = self.optimizer.param_groups[0]['lr']

        meta_model = copy.deepcopy(self.model)
        losses = self.forward4loss(meta_model, inner_x, inner_y)
        loss = self.tw(losses).sum()

        grads = torch.autograd.grad(loss, meta_model.parameters())
        with torch.no_grad():
            for weight, grad in zip(meta_model.parameters(), grads):
                weight.copy_(weight - inner_lr * grad)
        del grads, losses, loss

        losses = self.forward4loss(meta_model, outer_x, outer_y)
        loss = losses.sum()
        
        # compute hessian via finite difference approximation
        grads = torch.autograd.grad(loss, meta_model.parameters(), allow_unused=True)
        hessian = compute_hessian(self, grads, inner_x, inner_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip(self.tw.parameters(), hessian):
                mw.grad = - inner_lr * h
        self.tw_optimizer.step()

        del meta_model, losses, loss, grads, hessian

        # update model
        if not self.multi_input:
            all_x, all_y = self._process_data(org_train_dataloaders)
        else:
            all_x, all_y = {}, {}
            for task in self.task_name:
                each_x, each_y = self._process_data(org_train_dataloaders[task])
                all_x[task], all_y[task] = each_x, each_y

        losses, train_preds = self.forward4loss(self.model, all_x, all_y, return_preds=True)
        if not self.multi_input:
            self.meter.update(train_preds, all_y)
        else:
            for tn, task in enumerate(self.task_name):
                self.meter.update(train_preds[task], all_y[task], task)
        loss = self.tw(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.tw_optimizer.zero_grad()


    def bacth_forward_MOML(self, inner_x, inner_y, outer_x, outer_y, org_train_dataloaders):
        r"""Multi-Objective Meta Learning (MOML)

        This method is proposed in `Multi-Objective Meta Learning (NeurIPS 2021; AIJ 2024) <https://proceedings.neurips.cc/paper/2021/hash/b23975176653284f1f7356ba5539cfcb-Abstract.html>`_.
        """
        inner_lr = self.kwargs['weight_args']['inner_lr']
        assert self.kwargs['weight_args']['inner_step'] == 1, "This is a special implementation of MOML, which is fast but only supports the case of inner_step=1"

        # compute LL gradient g_f (tn x d)
        meta_model = copy.deepcopy(self.model)
        losses = self.forward4loss(meta_model, inner_x, inner_y)
        g_f = []
        for tn in range(self.task_num):
            g_f_tn = torch.autograd.grad(losses[tn], meta_model.parameters(), retain_graph=True)
            g_f.append(torch.cat([g.view(-1) for g in g_f_tn]))
        g_f = torch.stack(g_f)

        # update w* by one-step SGD
        beg = 0
        alpha = self.tw.weights.data.clone()
        for p in meta_model.parameters():
            p_grad = (F.softmax(alpha, dim=-1)@g_f[:, beg:beg+p.numel()]).view(p.size())
            p.data -= inner_lr * p_grad
            beg += p.numel()

        # compute softmax gradient
        g_s = torch.autograd.functional.jacobian(lambda x:F.softmax(x, dim=-1), alpha)

        # compute UL gradient g_F
        losses = self.forward4loss(meta_model, outer_x, outer_y)
        alpha_grad = torch.zeros(self.task_num, len(alpha)).to(self.device)
        for tn in range(self.task_num):
            g_F_tn = torch.autograd.grad(losses[tn], meta_model.parameters(), retain_graph=True)
            g_F_tn = torch.cat([g.view(-1) for g in g_F_tn]) # 1 x d
            alpha_grad[tn] = - inner_lr * g_F_tn @ g_s
        loss_data = torch.tensor([loss.item() for loss in losses]).to(self.device)
        from LibMTL.weighting.MGDA import MGDA
        MGDA_solver = MGDA()
        alpha_grad = MGDA_solver._gradient_normalizers(alpha_grad, loss_data, ntype='l2') # l2, loss, loss+, none
        sol = MGDA_solver._find_min_norm_element(alpha_grad)
        alpha_grad = sum([sol[tn] * alpha_grad[tn] for tn in range(self.task_num)])
        del meta_model

        self.tw_optimizer.zero_grad()
        for param in self.tw.parameters():
            param.grad = alpha_grad
        self.tw_optimizer.step()

        # update model
        if not self.multi_input:
            all_x, all_y = self._process_data(org_train_dataloaders)
        else:
            all_x, all_y = {}, {}
            for task in self.task_name:
                each_x, each_y = self._process_data(org_train_dataloaders[task])
                all_x[task], all_y[task] = each_x, each_y

        losses, train_preds = self.forward4loss(self.model, all_x, all_y, return_preds=True)
        if not self.multi_input:
            self.meter.update(train_preds, all_y)
        else:
            for tn, task in enumerate(self.task_name):
                self.meter.update(train_preds[task], all_y[task], task)

        loss = self.tw(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.tw_optimizer.zero_grad()


    def bacth_forward_FORUM(self, inner_x, inner_y, outer_x, outer_y, lambda_buffer, epoch):
        r'''FORUM
    
        This method is proposed in `A First-Order Multi-Gradient Algorithm for Multi-Objective Bi-Level Optimization (ECAI 2024) <https://ebooks.iospress.nl/doi/10.3233/FAIA240793>`_.        
        '''

        phi = self.kwargs['weight_args']['FORUM_phi']
        inner_step = self.kwargs['weight_args']['inner_step']
        inner_lr = self.kwargs['weight_args']['inner_lr']

        grad_index = []
        for param in self.model.parameters():
            grad_index.append(param.data.numel())
        
        # LL f(alpha, omega)
        train_losses, train_preds = self.forward4loss(self.model, inner_x, inner_y, return_preds=True)
        if not self.multi_input:
            self.meter.update(train_preds, inner_y)
        else:
            for tn, task in enumerate(self.task_name):
                self.meter.update(train_preds[task], inner_y[task], task)
        loss = self.tw(train_losses)
        # grad from f(alpha, omega)
        g_f = torch.autograd.grad(loss, list(self.tw.parameters())+list(self.model.parameters()))
        g_f = torch.cat([g.view(-1) for g in g_f])

        # f(alpha, omega^T)
        inner_model = copy.deepcopy(self.model)
        inner_optim = torch.optim.SGD(inner_model.parameters(), lr=inner_lr, weight_decay=0)
        for i in range(inner_step):
            train_losses = self.forward4loss(inner_model, inner_x, inner_y)
            loss = self.tw(train_losses)
            if loss.item() > 1e+5: # caused by too large inner_lr and inner_step
                break
            inner_optim.zero_grad()
            loss.backward()
            inner_optim.step()
            self.tw_optimizer.zero_grad()

        # grad from f^hat(alpha, omega^T)
        train_losses = self.forward4loss(inner_model, inner_x, inner_y)
        loss = self.tw(train_losses)
        g_f_hat_alpha = torch.autograd.grad(loss, self.tw.parameters())
        g_f_hat_alpha = torch.cat([g.view(-1) for g in g_f_hat_alpha])

        # grad from q_beta
        g_q_beta = copy.deepcopy(g_f)
        g_q_beta[:self.task_num] = g_q_beta[:self.task_num] - g_f_hat_alpha # size: [d]

        # F(omega)
        train_losses = self.forward4loss(self.model, outer_x, outer_y)
        g_F_omega_list = []
        for tn, task in enumerate(self.task_name):
            g_F_omega_tn = torch.autograd.grad(train_losses[tn], self.model.parameters(), retain_graph=True)
            g_F_omega_list.append(torch.cat([g.view(-1) for g in g_F_omega_tn]))
        g_F_omega_list = torch.stack(g_F_omega_list)
        # normalize g_F_omega_list
        gn = g_F_omega_list.pow(2).sum(-1).sqrt()
        g_F_omega_list = g_F_omega_list / gn.unsqueeze(1).repeat(1, g_F_omega_list.size()[1])

        g_F_omega_list = torch.cat([torch.zeros(self.task_num, self.task_num).to(self.device), g_F_omega_list], dim=1)

        # pi
        pi = []
        for tn in range(self.task_num):
            pi.append(phi - ((g_q_beta*g_F_omega_list[tn]).sum()/(g_q_beta.norm().pow(2)+1e-8)).item())

        w_constant = phi * (g_q_beta.norm().pow(2)+1e-8).item()

        # A
        A = torch.cat([g_F_omega_list, g_q_beta.unsqueeze(0)], dim=0) # (task_num+1) x d
        AAT = (A @ A.t()).detach().cpu().numpy()

        c, v = np.linalg.eig(AAT)
        # print(c, v)
        gg_sqrt = v @ np.diag(np.sqrt(np.maximum(c,0))) @ np.linalg.inv(v)
        # print(gg_sqrt)
        g_cp = cp.Parameter(shape=(self.task_num+1, self.task_num+1), value=gg_sqrt)
        w_cp = cp.Variable(shape=(self.task_num+1), nonneg=True)
        constraints = [cp.sum(w_cp[:-1]) == 1, 
                       w_cp[-1] >= cp.sum([w_cp[tn]*pi[tn] for tn in range(self.task_num)]), 
                       w_cp >= 0]
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(g_cp @ w_cp, 1) - w_cp[-1] * w_constant), constraints)
        prob.solve()
        w_cpu = w_cp.value

        # EMA lambda
        for tn in range(self.task_num):
            lambda_buffer[tn] = lambda_buffer[tn] + (1 / (epoch+1)**(3/4)) * (w_cpu[tn] - lambda_buffer[tn])

        nu = max(sum([lambda_buffer[tn]*pi[tn] for tn in range(self.task_num)]), 0)
        g_final = sum([lambda_buffer[tn]*g_F_omega_list[tn] for tn in range(self.task_num)]) + nu*g_q_beta

        self.tw_optimizer.zero_grad()
        for param in self.tw.parameters():
            param.grad = g_final[:self.task_num]

        self.optimizer.zero_grad()
        count = 0
        for param in self.model.parameters():
            beg = 0 if count == 0 else sum(grad_index[:count])
            end = sum(grad_index[:(count+1)])
            param.grad = g_final[self.task_num:][beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1

        self.tw_optimizer.step()
        self.optimizer.step()

        return lambda_buffer

    def train_evo_using_gaussian(self, train_dataloaders, test_dataloaders, epochs, 
                  val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        print(list(self.task_dict.keys()))

        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.evaluator import Evaluator
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.mutation.gauss import GaussianMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.operators.crossover.pntx import TwoPointCrossover
        from pymoo.core.problem import Problem
        from pymoo.core.termination import NoTermination
        from pymoo.problems.static import StaticProblem
        from pymoo.indicators.hv import HV
        from pymoo.visualization.scatter import Scatter

        hv_indicator = HV(ref_point=np.ones(self.task_num)+0.1)

        NP = self.kwargs['weight_args']['EVO_pop_size']
        W_init = self.kwargs['weight_args']['EVO_ws']
        initial_weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
        best_weights = initial_weights
        total_weights = len(initial_weights)
        # codebook, centers, log_sigmas, assignment = ubp_cluster(W=W_init, params=best_weights.detach().cpu().numpy())
        # W = len(centers)
        # D = W * 2
        # del centers, log_sigmas, assignment
        algorithm = NSGA2(pop_size=NP, mutation=PolynomialMutation(eta=5))
        # # D = None
        # problem = Problem(n_var=D, n_obj=3, n_constr=0, xl=np.ones(D) * -1, xu=np.ones(D))
        # # let the algorithm object never terminate and let the loop control it
        termination = NoTermination()
        # # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        # algorithm.setup(problem, termination=termination, seed=1, verbose=False)
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.meter.record_time('begin')
            reset = False
            if epoch % 10 == 0:
                # self.model = copy.deepcopy(ea_model)
                torch.nn.utils.vector_to_parameters(best_weights, self.model.parameters())
                self.model.train()
                for i, batch_index in enumerate(range(train_batch)):
                    train_losses = []
                    for sample_num in range(3 if self.weighting in ['MoDo', 'SDMGrad'] else 1):
                        if not self.multi_input:
                            train_inputs, train_gts = self._process_data(train_loader)
                            # print(f"data shape: {train_inputs.shape}")
                        else:
                            train_inputs, train_gts = {}, {}
                            for tn, task in enumerate(self.task_name):
                                train_input, train_gt = self._process_data(train_loader[task])
                                train_inputs[task], train_gts[task] = train_input, train_gt
                                # print(f"task: {task}, data shape: {train_inputs[task].shape}")
                        train_losses_, train_preds = self.forward4loss(self.model, train_inputs, train_gts, return_preds=True)
                        train_losses.append(train_losses_)

                    train_losses = torch.stack(train_losses).squeeze(0)
                    if not self.multi_input:
                            self.meter.update(train_preds, train_gts)
                    else:
                        for tn, task in enumerate(self.task_name):
                            self.meter.update(train_preds[task], train_gts[task], task)

                    self.optimizer.zero_grad(set_to_none=False)
                    w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                    if w is not None:
                        self.batch_weight[:, epoch, batch_index] = w
                    self.optimizer.step()

                    # if i == 0:
                    #     break

                self.meter.record_time('end')
                self.meter.get_score()
                self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
                self.meter.display(epoch=epoch, mode='train')
                self.meter.reinit()
                gd_loss = self.forward4loss(self.model, train_inputs, train_gts)
                print(f"gd_loss: {gd_loss.detach().cpu().numpy()}")
                gd_val_loss = self.test(test_dataloaders, epoch, mode='test', num_batch=None)

                best_weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
                codebook, centers, log_sigmas, assignment = ubp_cluster(W=W_init, params=best_weights.detach().cpu().numpy())
                W = len(centers)
                D = W * 2
                print(f"W: {W}, D: {D}")
                x0 = np.concatenate([centers, log_sigmas])
                init_population = np.zeros((NP, D))
                init_fitness = np.full(NP, np.inf)
                for i in range(NP):
                    for j in range(W):
                        init_population[i][j] = np.random.uniform(centers[j] - np.exp(log_sigmas[j])/6, centers[j] + np.exp(log_sigmas[j])/6)
                        init_population[i][j+W] = np.random.normal(log_sigmas[j], 0.001)
                init_population[-1] = x0.copy()
                problem = Problem(n_var=D, n_obj=self.task_num, n_constr=0, xl=np.ones(D) * x0.min(), xu=np.ones(D) * x0.max())
                # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
                algorithm = NSGA2(pop_size=NP, mutation=GaussianMutation(sigma=0.001), crossover=UniformCrossover(prob=0.5))
                algorithm.setup(problem, termination=termination, seed=1, verbose=False)
                # ea_model = copy.deepcopy(self.model)
                reset = True

            if not self.multi_input:
                train_inputs, train_gts = self._process_data(train_loader)
            else:
                train_inputs, train_gts = {}, {}
                for tn, task in enumerate(self.task_name):
                    train_input, train_gt = self._process_data(train_loader[task])
                    train_inputs[task], train_gts[task] = train_input, train_gt

            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()
            
            # get the design space values of the algorithm
            X = pop.get("X")
            if reset:
                pop.set("X", init_population)
                X = pop.get("X")
            # evaluate the solution
            losses = np.zeros((NP, self.task_num))
            for i in range(NP):
                x = X[i]
                build_model_from_blocks_using_gaussian(self.model, W, total_weights, x, codebook, None, None)
                # ea_model.eval()
                with torch.no_grad():
                    losses[i] = self.forward4loss(self.model, train_inputs, train_gts).detach().cpu().numpy()
                    # print(losses[i])

            F = np.column_stack(losses)

            static = StaticProblem(problem, F=F)
            Evaluator().eval(static, pop)

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

            res = algorithm.result()
            pf_F = res.F
            pop_F = res.pop.get("F")
            plot_pareto_front_and_population(pf_F=pf_F, pop_F=pop_F, gd_point=gd_loss.detach().cpu().numpy(), iter=epoch, save_path=self.save_path+"/train/", loss_names=list(self.task_dict.keys()))

            pf_val_losses = np.zeros((len(pf_F), self.task_num))
            for i in range(len(pf_F)):
                x = res.X[i]
                build_model_from_blocks_using_gaussian(self.model, W, total_weights, x, codebook, None, None)
                pf_val_losses[i] = self.test(test_dataloaders, epoch, mode='test', num_batch=None).detach().cpu().numpy()

            plot_pareto_front_and_population(pf_F=pf_val_losses, pop_F=None, gd_point=gd_val_loss.detach().cpu().numpy(), iter=epoch, save_path=self.save_path+"/val/", loss_names=list(self.task_dict.keys()))

            avg_rank_losses = np.mean(np.argsort(res.F, axis=1), axis=1)
            build_model_from_blocks_using_gaussian(self.model, W, total_weights, 
                                               np.mean(res.X[avg_rank_losses.argsort()[:1]], axis=0), 
                                               codebook, None, None)
            
            curr_best_weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
            # if hv_indicator.do(res.F) > hv_indicator.do(gd_loss.detach().cpu().numpy().reshape(1, -1)):
            if True:
                best_weights = curr_best_weights

            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            # if self.scheduler is not None:
            #     self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))


            
            
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight



    def train_evo_using_centers(self, train_dataloaders, test_dataloaders, epochs, 
                  val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        # print(list(self.task_dict.keys()))

        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.evaluator import Evaluator
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.mutation.gauss import GaussianMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.operators.crossover.pntx import TwoPointCrossover
        from pymoo.core.problem import Problem
        from pymoo.core.termination import NoTermination
        from pymoo.problems.static import StaticProblem
        from pymoo.indicators.hv import HV
        from pymoo.visualization.scatter import Scatter

        hv_indicator = HV(ref_point=np.ones(self.task_num)+0.01)

        NP = self.kwargs['weight_args']['EVO_pop_size']
        W_init = self.kwargs['weight_args']['EVO_ws']
        initial_weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
        best_weights = initial_weights
        total_weights = len(initial_weights)
        # codebook, centers, log_sigmas, assignment = ubp_cluster(W=W_init, params=best_weights.detach().cpu().numpy())
        # W = len(centers)
        # D = W * 2
        # del centers, log_sigmas, assignment
        algorithm = NSGA2(pop_size=NP, mutation=PolynomialMutation(eta=5))
        # # D = None
        # problem = Problem(n_var=D, n_obj=3, n_constr=0, xl=np.ones(D) * -1, xu=np.ones(D))
        # # let the algorithm object never terminate and let the loop control it
        termination = NoTermination()
        # # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        # algorithm.setup(problem, termination=termination, seed=1, verbose=False)
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.meter.record_time('begin')
            reset = False
            if epoch % 10 == 0:
                # self.model = copy.deepcopy(ea_model)
                torch.nn.utils.vector_to_parameters(best_weights, self.model.parameters())
                self.model.train()
                gd_loss = 0
                for i, batch_index in enumerate(range(train_batch)):
                    train_losses = []
                    for sample_num in range(3 if self.weighting in ['MoDo', 'SDMGrad'] else 1):
                        if not self.multi_input:
                            train_inputs, train_gts = self._process_data(train_loader)
                            # print(f"data shape: {train_inputs.shape}")
                        else:
                            train_inputs, train_gts = {}, {}
                            for tn, task in enumerate(self.task_name):
                                train_input, train_gt = self._process_data(train_loader[task])
                                train_inputs[task], train_gts[task] = train_input, train_gt
                                # print(f"task: {task}, data shape: {train_inputs[task].shape}")
                        train_losses_, train_preds = self.forward4loss(self.model, train_inputs, train_gts, return_preds=True)
                        train_losses.append(train_losses_)

                    train_losses = torch.stack(train_losses).squeeze(0)
                    if not self.multi_input:
                            self.meter.update(train_preds, train_gts)
                    else:
                        for tn, task in enumerate(self.task_name):
                            self.meter.update(train_preds[task], train_gts[task], task)

                    self.optimizer.zero_grad(set_to_none=False)
                    w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                    if w is not None:
                        self.batch_weight[:, epoch, batch_index] = w
                    self.optimizer.step()
                    gd_loss += train_losses.detach().cpu().numpy()

                    # if i == 0:
                    #     break
                gd_loss /= train_batch
                
                print(f"********** SGD Evaluation **********")
                self.meter.record_time('end')
                self.meter.get_score()
                self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
                self.meter.display(epoch=epoch, mode='train')
                self.meter.reinit()
                
                # gd_loss = self.forward4loss(self.model, train_inputs, train_gts)
                # print(f"train gd loss: {gd_loss.detach().cpu().numpy()}")
                gd_val_loss, gd_val_results = self.test(test_dataloaders, epoch, mode='test', num_batch=None, verbose=False)
                print(gd_val_loss)
                # gd_val_results['avg'] = [np.mean([res for res in gd_val_results.values()])]
                print(pd.DataFrame(gd_val_results))


                best_weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
                codebook, centers, log_sigmas, assignment = ubp_cluster(W=W_init, params=best_weights.detach().cpu().numpy())
                W = len(centers)
                D = W
                print(f"W: {W}, D: {D}")
                x0 = centers
                init_population = np.zeros((NP, D))
                init_fitness = np.full(NP, np.inf)
                for i in range(NP):
                    for j in range(W):
                        init_population[i][j] = np.random.uniform(centers[j] - np.exp(log_sigmas[j])/6, centers[j] + np.exp(log_sigmas[j])/6)
                init_population[-1] = x0.copy()
                problem = Problem(n_var=D, n_obj=self.task_num, n_constr=0, xl=np.ones(D) * x0.min(), xu=np.ones(D) * x0.max())
                # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
                algorithm = NSGA2(pop_size=NP, mutation=GaussianMutation(sigma=0.001), crossover=UniformCrossover(prob=0.5))
                algorithm.setup(problem, termination=termination, seed=1, verbose=False)
                # ea_model = copy.deepcopy(self.model)
                reset = True

            if not self.multi_input:
                train_inputs, train_gts = self._process_data(train_loader)
            else:
                train_inputs, train_gts = {}, {}
                for tn, task in enumerate(self.task_name):
                    train_input, train_gt = self._process_data(train_loader[task])
                    train_inputs[task], train_gts[task] = train_input, train_gt

            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()
            
            # get the design space values of the algorithm
            X = pop.get("X")
            if reset:
                pop.set("X", init_population)
                X = pop.get("X")
            # evaluate the solution
            losses = np.zeros((NP, self.task_num))
            for i in range(NP):
                x = X[i]
                build_model_from_blocks_using_centers(self.model, W, total_weights, x, codebook, None, None)
                self.model.eval()
                with torch.no_grad():
                    losses[i] = self.forward4loss(self.model, train_inputs, train_gts).detach().cpu().numpy()

            F = np.column_stack(losses)
            static = StaticProblem(problem, F=F)
            Evaluator().eval(static, pop)

            if len(algorithm.pop.get("X")) > 0:
                X = algorithm.pop.get("X")
                F = algorithm.pop.get("F")
                for i in range(NP):
                    x = X[i]
                    build_model_from_blocks_using_centers(self.model, W, total_weights, x, codebook, None, None)
                    self.model.eval()
                    with torch.no_grad():
                        F[i] = self.forward4loss(self.model, train_inputs, train_gts).detach().cpu().numpy()
                algorithm.pop.set("F", F)
            
            algorithm.tell(infills=pop)

            res = algorithm.result()
            pf_F = res.F
            pop_F = res.pop.get("F")
            plot_pareto_front_and_population(pf_F=pf_F, pop_F=pop_F, gd_point=gd_loss, iter=epoch, save_path=self.save_path+"/train/", loss_names=list(self.task_dict.keys()))

            pf_val_losses = np.zeros((len(pf_F), self.task_num))
            pf_val_results = []
            print(f"********** EVO Pareto Front Evaluation **********")
            for i in range(len(pf_F)):
                x = res.X[i]
                build_model_from_blocks_using_centers(self.model, W, total_weights, x, codebook, None, None)
                losses, results = self.test(test_dataloaders, epoch, mode='test', num_batch=None, verbose=False)
                pf_val_losses[i] = losses
                pf_val_results.append(results)
            # print(pf_val_losses)
            print(pd.DataFrame(pf_val_results))

            plot_pareto_front_and_population(pf_F=pf_val_losses, pop_F=None, gd_point=gd_val_loss, iter=epoch, save_path=self.save_path+"/val/", loss_names=list(self.task_dict.keys()))

            # avg_rank_losses = np.mean(np.argsort(res.F, axis=1), axis=1)
            avg_rank_losses = np.mean(res.F, axis=1)
            print(f"********** EVO Pareto Front Average Loss (on training) **********")
            # print(f"Pareto Front Average Loss: {avg_rank_losses}, Ranking: {avg_rank_losses.argsort()}")
            df = pd.DataFrame({
                "Average Loss": avg_rank_losses,
                "Ranking": pd.Series(avg_rank_losses).rank(method='min', ascending=True).astype(int)
            })
            print(df)
            print(f"Top solution: {avg_rank_losses.min()}, {np.argmin(avg_rank_losses)}")
            candidate = res.X[np.argmin(avg_rank_losses)]
            build_model_from_blocks_using_centers(self.model, W, total_weights, 
                                               candidate, 
                                               codebook, None, None)
            curr_best_weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
            # if hv_indicator.do(res.F) > hv_indicator.do(gd_loss.detach().cpu().numpy().reshape(1, -1)):
            if True:
                best_weights = curr_best_weights

            print(f"********** Best Solution Evaluation **********")
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            losses, results = self.test(test_dataloaders, epoch, mode='test')
            # print(losses)
            print(pd.DataFrame(results))
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            # if self.scheduler is not None:
            #     self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))


            
            
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight
