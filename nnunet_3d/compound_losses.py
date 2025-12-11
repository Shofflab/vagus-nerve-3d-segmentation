"""
Compound loss functions for nnU-Net v2.

This module provides custom loss functions combining Dice loss with cross-entropy
variants and topological interaction constraints for medical image segmentation.
"""

import torch
import numpy as np
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_and_CE_loss(nn.Module):
    """Combined Dice and Cross-Entropy loss for multi-class segmentation."""

    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Initialize the combined Dice and Cross-Entropy loss.

        Args:
            soft_dice_kwargs: Keyword arguments for the Dice loss.
            ce_kwargs: Keyword arguments for the Cross-Entropy loss.
            weight_ce: Weight for the Cross-Entropy loss component.
            weight_dice: Weight for the Dice loss component.
            ignore_label: Label value to ignore during loss computation.
            dice_class: Dice loss class to use (default: SoftDiceLoss).

        Note:
            Weights for CE and Dice do not need to sum to one.
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Compute the combined loss.

        Args:
            net_output: Network output of shape (B, C, X, Y[, Z]).
            target: Ground truth of shape (B, 1, X, Y[, Z]).

        Returns:
            Combined weighted loss value.
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # Replace ignored labels with 0 (arbitrary valid label since gradients are masked)
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    """Combined Dice and Binary Cross-Entropy loss for multi-label segmentation."""

    def __init__(
        self,
        bce_kwargs,
        soft_dice_kwargs,
        weight_ce=1,
        weight_dice=1,
        use_ignore_label: bool = False,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        """
        Initialize the combined Dice and BCE loss.

        Args:
            bce_kwargs: Keyword arguments for the BCE loss.
            soft_dice_kwargs: Keyword arguments for the Dice loss.
            weight_ce: Weight for the BCE loss component.
            weight_dice: Weight for the Dice loss component.
            use_ignore_label: Whether to use ignore labels. If True, the ignore
                mask is expected in target[:, -1].
            dice_class: Dice loss class to use (default: MemoryEfficientSoftDiceLoss).

        Note:
            Do not apply nonlinearity in your network when using this loss.
            Target must be one-hot encoded.
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs["reduction"] = "none"

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Compute the combined loss.

        Args:
            net_output: Network output logits of shape (B, C, X, Y[, Z]).
            target: One-hot encoded ground truth of shape (B, C, X, Y[, Z]).
                If use_ignore_label is True, the last channel contains the ignore mask.

        Returns:
            Combined weighted loss value.
        """
        if self.use_ignore_label:
            # Invert ignore channel to get valid region mask
            mask = (1 - target[:, -1:]).bool()
            # Remove ignore channel from target
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(
                mask.sum(), min=1e-8
            )
        else:
            ce_loss = self.ce(net_output, target_regions)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    """Combined Dice and Top-K Cross-Entropy loss for hard example mining."""

    def __init__(
        self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None
    ):
        """
        Initialize the combined Dice and Top-K loss.

        Args:
            soft_dice_kwargs: Keyword arguments for the Dice loss.
            ce_kwargs: Keyword arguments for the Top-K loss.
            weight_ce: Weight for the Top-K CE loss component.
            weight_dice: Weight for the Dice loss component.
            ignore_label: Label value to ignore during loss computation.

        Note:
            Weights for CE and Dice do not need to sum to one.
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Compute the combined loss.

        Args:
            net_output: Network output of shape (B, C, X, Y[, Z]).
            target: Ground truth of shape (B, 1, X, Y[, Z]).

        Returns:
            Combined weighted loss value.
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = (target != self.ignore_label).bool()
            # Replace ignored labels with 0 (arbitrary valid label since gradients are masked)
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target)
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class TI_Loss(nn.Module):
    """
    Topological Interaction (TI) Loss for enforcing spatial relationships.

    This loss penalizes topological violations between anatomical structures,
    such as inclusion (one structure must be inside another) and exclusion
    (two structures cannot overlap) constraints.

    Reference:
        Gupta et al., "Learning Topological Interactions for Multi-Class
        Medical Image Segmentation", ECCV 2022.
    """

    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        Initialize the Topological Interaction loss.

        Args:
            dim: Spatial dimensionality (2 for 2D, 3 for 3D).
            connectivity: Neighborhood connectivity.
                - 2D: 4 (cross) or 8 (square neighborhood)
                - 3D: 6 (face-connected) or 26 (cube neighborhood)
            inclusion: List of [A, B] class pairs where A must be inside B.
            exclusion: List of [A, C] class pairs where A and C must not touch.
            min_thick: Minimum thickness/separation between classes.
                Only used when connectivity is 8 (2D) or 26 (3D).
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        # Build interaction list: (is_inclusion, label_A, label_B/C)
        for inc in inclusion:
            self.interaction_list.append([True, inc[0], inc[1]])

        for exc in exclusion:
            self.interaction_list.append([False, exc[0], exc[1]])

    def set_kernel(self):
        """Initialize the connectivity kernel based on dim, connectivity, and min_thick."""
        k = 2 * self.min_thick + 1

        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))
        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array(
                    [
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    ]
                )
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

        self.kernel = torch.from_numpy(
            np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0)
        )

    def topological_interaction_module(self, P):
        """
        Compute the critical voxels map based on topological constraints.

        Args:
            P: Discrete segmentation map of shape (B, 1, X, Y[, Z]).

        Returns:
            Critical voxels map indicating topological violations.
        """
        critical_voxels_map = None

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Create binary masks for each class
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                # Inclusion: mask_C is everything except A and its required container B
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                # Exclusion: mask_C is the excluded class
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Compute neighborhood information via convolution
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding="same")
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding="same")
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Identify voxels that violate the topological constraint
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(
                    critical_voxels_map, violating
                ).double()

        return critical_voxels_map

    def forward(self, x, y):
        """
        Compute the Topological Interaction loss.

        Args:
            x: Network output logits of shape (B, C, X, Y[, Z]) where C is
                the number of classes.
            y: Ground truth of shape (B, 1, X, Y[, Z]) with values in [0, C).

        Returns:
            TI loss value penalizing topological violations.
        """
        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation from network output
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        # Compute critical voxels that violate topological constraints
        critical_voxels_map = self.topological_interaction_module(P)

        # Apply cross-entropy loss only at critical voxels
        ce_tensor = torch.unsqueeze(
            self.ce_loss_func(x.double(), y[:, 0].long()), dim=1
        )
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value


class DC_CE_TI_loss(nn.Module):
    """
    Combined Dice, Cross-Entropy, and Topological Interaction loss.

    This loss combines standard segmentation losses with topological constraints
    to encourage anatomically plausible segmentations.
    """

    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        weight_ti=1e-6,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Initialize the combined DC + CE + TI loss.

        Args:
            soft_dice_kwargs: Keyword arguments for the Dice loss.
            ce_kwargs: Keyword arguments for the Cross-Entropy loss.
            weight_ce: Weight for the Cross-Entropy loss component.
            weight_dice: Weight for the Dice loss component.
            weight_ti: Weight for the Topological Interaction loss component.
            ignore_label: Label value to ignore during loss computation.
            dice_class: Dice loss class to use (default: SoftDiceLoss).

        Note:
            Weights do not need to sum to one.
        """
        super(DC_CE_TI_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_ti = weight_ti
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ti = TI_Loss(
            dim=3, connectivity=26, inclusion=[[1, 2]], exclusion=[[0, 1]], min_thick=1
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Compute the combined loss.

        Args:
            net_output: Network output of shape (B, C, X, Y, Z).
            target: Ground truth of shape (B, 1, X, Y, Z).

        Returns:
            Combined weighted loss value.
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        ti_loss = self.ti(net_output, target)

        result = (
            self.weight_ce * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_ti * ti_loss
        )
        return result
