import torch
import torch.nn as nn
import torch.nn.functional as F

from ..reducers import AvgNonZeroReducer
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


def kl_div_gauss(mu, var, mu_prior, var_prior):
    """
    Compute KL divergence between two Gaussian distributions.
    
    Args:
        mu: mean of the first Gaussian
        var: variance of the first Gaussian
        mu_prior: mean of the prior Gaussian
        var_prior: variance of the prior Gaussian
    
    Returns:
        KL divergence
    """
    return 0.5 * torch.sum(
        torch.log(var_prior / var) + (var + (mu - mu_prior) ** 2) / var_prior - 1
    )


def kl_div_vMF(mu, var):
    """
    Compute KL divergence for von Mises-Fisher distribution.
    This is a simplified implementation - you may need to adjust based on your specific needs.
    
    Args:
        mu: mean direction
        var: concentration parameter
    
    Returns:
        KL divergence
    """
    # Simplified vMF KL divergence - you may need to implement the full version
    # based on your specific requirements
    return torch.sum(var * (1 - torch.norm(mu, dim=0)))


def negative_loglikelihood(mu_a, mu_p, mu_n, var_a, var_p, var_n, margin=0.05):
    """
    Compute negative log-likelihood for Bayesian triplet loss.
    
    Args:
        mu_a: anchor means
        mu_p: positive means
        mu_n: negative means
        var_a: anchor variances
        var_p: positive variances
        var_n: negative variances
        margin: margin for triplet loss
    
    Returns:
        Negative log-likelihood
    """
    # Compute distances
    ap_dist = torch.sum((mu_a - mu_p) ** 2, dim=0)
    an_dist = torch.sum((mu_a - mu_n) ** 2, dim=0)
    
    # Add variance terms
    ap_var = torch.sum(var_a + var_p, dim=0)
    an_var = torch.sum(var_a + var_n, dim=0)
    
    # Compute log-likelihood
    log_prob_positive = -0.5 * (ap_dist / ap_var + torch.log(2 * torch.pi * ap_var))
    log_prob_negative = -0.5 * (an_dist / an_var + torch.log(2 * torch.pi * an_var))
    
    # Triplet constraint: ap_dist + margin < an_dist
    violation = ap_dist + margin - an_dist
    triplet_loss = F.relu(violation)
    
    # Combine log-likelihood and triplet constraint
    nll = -torch.mean(log_prob_positive) + torch.mean(triplet_loss)
    
    return nll


class BayesianTripletLoss(BaseMetricLossFunction):
    """
    Bayesian Triplet Loss that incorporates uncertainty in embeddings.
    
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        var_prior: Prior variance for KL divergence regularization
        kl_scale_factor: Scaling factor for KL divergence term
        distribution: Distribution type for KL divergence ('gauss' or 'vMF')
        triplets_per_anchor: Number of triplets per anchor
    """

    def __init__(
        self,
        margin=0.05,
        var_prior=1.0,
        kl_scale_factor=1e-6,
        distribution='gauss',
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.var_prior = var_prior
        self.kl_scale_factor = kl_scale_factor
        self.distribution = distribution
        self.triplets_per_anchor = triplets_per_anchor
        
        self.add_to_recordable_attributes(
            list_of_names=["margin", "var_prior", "kl_scale_factor", "distribution"], 
            is_stat=False
        )

    def compute_loss(self, embeddings, labels, indices_tuple=None):
        """
        Compute the Bayesian triplet loss.
        
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
        
        Returns:
            Dictionary containing loss information
        """
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        
        if len(anchor_idx) == 0:
            return self.zero_losses()

        # Split embeddings into means and variances
        # Assuming the last dimension contains variance information
        embedding_dim = embeddings.size(1)
        if embedding_dim % 2 == 0:
            # Even dimension: first half is mean, second half is variance
            mu_dim = embedding_dim // 2
            mu_embeddings = embeddings[:, :mu_dim]
            var_embeddings = F.softplus(embeddings[:, mu_dim:])  # Ensure positive variance
        else:
            # Odd dimension: all but last are mean, last is variance
            mu_embeddings = embeddings[:, :-1]
            var_embeddings = F.softplus(embeddings[:, -1:])  # Ensure positive variance

        # Extract anchor, positive, and negative embeddings
        mu_a = mu_embeddings[anchor_idx]
        mu_p = mu_embeddings[positive_idx]
        mu_n = mu_embeddings[negative_idx]
        
        var_a = var_embeddings[anchor_idx]
        var_p = var_embeddings[positive_idx]
        var_n = var_embeddings[negative_idx]

        # Calculate negative log-likelihood
        nll = negative_loglikelihood(
            mu_a.T, mu_p.T, mu_n.T, var_a.T, var_p.T, var_n.T, margin=self.margin
        )

        # Calculate KL divergence
        kl = torch.tensor(0.0, device=embeddings.device)

        if self.distribution == 'gauss':
            mu_prior = torch.zeros_like(mu_a, requires_grad=False)
            var_prior_tensor = torch.ones_like(var_a, requires_grad=False) * self.var_prior

            kl = (kl_div_gauss(mu_a, var_a, mu_prior, var_prior_tensor) + 
                  kl_div_gauss(mu_p, var_p, mu_prior, var_prior_tensor) + 
                  kl_div_gauss(mu_n, var_n, mu_prior, var_prior_tensor))

        elif self.distribution == 'vMF':
            kl = (kl_div_vMF(mu_a, var_a) + 
                  kl_div_vMF(mu_p, var_p) + 
                  kl_div_vMF(mu_n, var_n))

        # Total loss
        total_loss = nll + self.kl_scale_factor * kl

        return {
            "loss": {
                "losses": total_loss,
                "indices": indices_tuple,
                "reduction_type": "already_reduced",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')' 