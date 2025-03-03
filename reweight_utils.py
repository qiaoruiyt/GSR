from torch_influence.base import BaseInfluenceModule, BaseObjective
from torch import nn
from torch.utils import data
import torch
from typing import List, Optional
import numpy as np
import logging
from torch.func import functional_call, vmap, grad

class GradOnlyInfluenceModule(BaseInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

    def inverse_hvp(self, vec):
        # simply remove the Hessian
        return vec


class FastExactInfluenceModule(BaseInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            check_eigvals: bool = False,
            grad_bs: int = 1000,
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.grad_bs = grad_bs

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        d = flat_params.shape[0]
        hess = 0.0

        for batch, batch_size in self._loader_wrapper(train=True):
            def f(theta_):
                self._model_reinsert_params(self._reshape_like_params(theta_))
                return self.objective.train_loss(self.model, theta_, batch)

            hess_batch = torch.autograd.functional.hessian(f, flat_params).detach()
            hess = hess + hess_batch * batch_size

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
            hess = hess / len(self.train_loader.dataset)
            hess = hess + damp * torch.eye(d, device=hess.device)

            if check_eigvals:
                eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()

            self.inverse_hess = torch.inverse(hess)

        grads = []
        train_idxs = list(range(len(self.train_loader.dataset)))
        for grad_z, _ in self.per_sample_loss_grad_loader_wrapper(batch_size=self.grad_bs, subset=train_idxs, train=True):
            grads.append(grad_z)
        train_grads = torch.cat(grads, dim=0)
        self.train_vihp = train_grads @ self.inverse_hess

    def inverse_hvp(self, vec):
        raise Exception("This should not be called")
        return self.inverse_hess @ vec

    def influences(
            self,
            train_idxs: List[int],
            test_idxs: List[int],
            stest: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if stest is not None:
            raise Exception("stest is not supported")
        test_grads = self.test_loss_grad(test_idxs)
        return self.train_vihp[train_idxs, :] @ test_grads / len(self.train_loader.dataset)

    def per_sample_loss_grad_loader_wrapper(self, train, **kwargs):
        if train:
            if hasattr(self.objective, "sample_weights") and self.objective.sample_weights is not None:
                loss_fn = self.objective.unweighted_train_loss
            else:
                loss_fn = self.objective.train_loss
        else:
            loss_fn = self.objective.test_loss

        model = self.model
        params_dict = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        def compute_loss(params_dict, buffers, batch):
            model = lambda batch: functional_call(self.model, (params_dict, buffers), (batch,))
            flat_params = [v for _, v in params_dict.items()]
            flat_params = self._flatten_params_like(flat_params)
            loss = loss_fn(model, flat_params, batch)
            return loss

        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
        
        for batch, batch_size in self._loader_wrapper(train=train, **kwargs):
            ft_per_sample_grads = ft_compute_sample_grad(params_dict, buffers, batch)
            param_grads = [grad.view(grad.size(0), -1) for _, grad in ft_per_sample_grads.items()]
            param_grads = torch.cat(param_grads, dim=1)
            yield param_grads, batch_size


class FastHFInfluenceModule(BaseInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            check_eigvals: bool = False,
            grad_bs: int = 1000,
            damp: float = 0.0,
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.grad_bs = grad_bs

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        d = flat_params.shape[0]

        grads = []
        train_idxs = list(range(len(self.train_loader.dataset)))
        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
        for grad_z, _ in self.per_sample_loss_grad_loader_wrapper(batch_size=self.grad_bs, subset=train_idxs, train=True):
            grads.append(grad_z)
        train_grads = torch.cat(grads, dim=0)
        self.train_vihp = train_grads 

    def inverse_hvp(self, vec):
        raise Exception("This should not be called")
        return self.inverse_hess @ vec

    def influences(
            self,
            train_idxs: List[int],
            test_idxs: List[int],
            stest: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if stest is not None:
            raise Exception("stest is not supported")
        test_grads = self.test_loss_grad(test_idxs)
        return self.train_vihp[train_idxs, :] @ test_grads / len(self.train_loader.dataset)

    def per_sample_loss_grad_loader_wrapper(self, train, **kwargs):
        if train:
            if hasattr(self.objective, "sample_weights") and self.objective.sample_weights is not None:
                loss_fn = self.objective.unweighted_train_loss
            else:
                loss_fn = self.objective.train_loss
        else:
            loss_fn = self.objective.test_loss

        model = self.model
        params_dict = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        def compute_loss(params_dict, buffers, batch):
            model = lambda batch: functional_call(self.model, (params_dict, buffers), (batch,))
            flat_params = [v for _, v in params_dict.items()]
            flat_params = self._flatten_params_like(flat_params)
            loss = loss_fn(model, flat_params, batch)
            return loss

        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
        
        for batch, batch_size in self._loader_wrapper(train=train, **kwargs):
            ft_per_sample_grads = ft_compute_sample_grad(params_dict, buffers, batch)
            param_grads = [grad.view(grad.size(0), -1) for _, grad in ft_per_sample_grads.items()]
            param_grads = torch.cat(param_grads, dim=1)
            yield param_grads, batch_size

