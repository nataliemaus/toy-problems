from abc import ABC, abstractmethod
from typing import Any, Optional
from unittest import runner

import torch
from botorch.acquisition.objective import (
    AcquisitionObjective,
    IdentityMCObjective,
    ScalarizedObjective,
)
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module


# Code copied form botorch MaxPosteriorSampling class 
# https://botorch.org/api/_modules/botorch/generation/sampling.html


class SamplingStrategy(Module, ABC):
    r"""Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1, **kwargs: Any) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


# MaxPosteriorSampling class from botorch modified to support sampling with constraints
class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5) 
    """
    def __init__(
        self,
        models_list: list = None,
        objective: Optional[AcquisitionObjective] = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The objective. Typically, the AcquisitionObjective under which
                the samples are evaluated. If a ScalarizedObjective, samples from the
                scalarized posterior are used. Defaults to `IdentityMCObjective()`.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.models_list = models_list
        if objective is None:
            objective = IdentityMCObjective()
        self.objective = objective
        self.replacement = replacement

    def forward(
        self,
        ys_to_scores_objective,
        X: Tensor, 
        num_samples: int = 1, 
        observation_noise: bool = False, 
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        all_y_samples = []
        for model in self.models_list:
            posterior = model.posterior(X, observation_noise=observation_noise)
            if isinstance(self.objective, ScalarizedObjective):
                posterior = self.objective(posterior)
            y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
            all_y_samples.append(y_samples)
        # all_y_samples[0].shape = torch.Size([10, 2400, 1]) = (bsz/num_samples, N, 1)
        all_y_samples = torch.cat(all_y_samples, dim=-1)  # samples = bsz x N x output_dim ? 
        # all_y_samples.shape = torch.Size([10, 2400, 8]) = (bsz, N, output_dim)
        all_y_samples = all_y_samples.reshape(-1, all_y_samples.shape[-1]) 
        # all_y_samples.shape = torch.Size([24000, 8]) = (bsz*N, output_dim)
        all_s_samples = ys_to_scores_objective.ys_to_scores(all_y_samples)
        # all_s_samples.shape = torch.Size([24000, 1]) = (bsz*N, 1)
        samples = all_s_samples.reshape(num_samples, -1, 1).cuda() 
        # samples.shape = torch.Size([10, 2400, 1]) = (bsz, N, 1 )  ... YAY 
        
        if isinstance(self.objective, ScalarizedObjective):
            obj = samples.squeeze(-1)  # num_samples x batch_shape x N 
        else:
            obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs) # means use the idcs to index dimension -2 