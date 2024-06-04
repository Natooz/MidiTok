"""Collator objects for PyTorch ``DataLoader``s."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from torch import LongTensor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class DataCollator:
    r"""
    All-in-one data collator for PyTorch ``DataLoader``.

    It allows to apply padding (right or left side of sequences), prepend or append
    *BOS* and *EOS* tokens. It will also add an ``"attention_mask"`` entry to the
    batch, following the padding applied.

    :param pad_token_id: padding token id.
    :param pad_on_left: if given True, it will pad the sequences on the left. This
        can be required when using some libraries expecting padding on left, for
        example when generating with Hugging Face Transformers. (default: ``False``)
    :param copy_inputs_as_labels: will add a labels entry (``inputs_kwarg_name``) to
        the batch (or replace the existing one), which is a copy to the input entry
        (``labels_kwarg_name``). (default: ``False``)
    :param shift_labels: will shift inputs and labels for autoregressive
        training/teacher forcing. (default: ``False``)
    :param labels_pad_idx: padding id for labels. (default: -100)
    :param inputs_kwarg_name: name of dict / kwarg key for inputs.
        (default: ``"input_ids"``)
    :param labels_kwarg_name: name of dict / kwarg key for inputs.
        (default: ``"labels"``)
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_on_left: bool = False,
        copy_inputs_as_labels: bool = False,
        shift_labels: bool = False,
        labels_pad_idx: int = -100,
        inputs_kwarg_name: str = "input_ids",
        labels_kwarg_name: str = "labels",
    ) -> None:
        self.pad_token = pad_token_id
        self.pad_on_left = pad_on_left
        self.copy_inputs_as_labels = copy_inputs_as_labels
        self.shift_labels = shift_labels
        self.labels_pad_idx = labels_pad_idx
        self.inputs_kwarg_name = inputs_kwarg_name
        self.labels_kwarg_name = labels_kwarg_name

    def __call__(self, batch: list[Mapping[str, Any]]) -> Mapping[str, LongTensor]:
        """
        Collate the sequences of a batch, make them ready to be fed to a model.

        :param batch: batch of sequences, as a list of dictionaries containing input ids
            and optionally labels.
        :return: the output batch as a dictionary linking to input and optionally target
            tensors.
        """
        out_batch = {}
        x, y = None, None

        # Figure out inputs
        if self.inputs_kwarg_name in batch[0]:
            x = [
                sample[self.inputs_kwarg_name]
                for sample in batch
                if sample[self.inputs_kwarg_name] is not None
                and len(sample[self.inputs_kwarg_name]) > 0
            ]

        # Figure out labels
        if self.labels_kwarg_name in batch[0]:
            y = [
                sample[self.labels_kwarg_name]
                for si, sample in enumerate(batch)
                if sample[self.labels_kwarg_name] is not None
                and len(batch[si][self.inputs_kwarg_name]) > 0  # x
            ]
        elif self.copy_inputs_as_labels:
            y = deepcopy(x)

        # Pad inputs / convert to Tensors
        if x is not None:
            x = _pad_batch(x, self.pad_token, self.pad_on_left)
        if y is not None:
            # If labels are sequences of tokens
            if y[0].dim() > 0:
                y = _pad_batch(y, self.labels_pad_idx, self.pad_on_left)
            else:  # classification
                y = torch.stack(y)

        # Shift labels, otherwise it's handled by models
        if self.shift_labels:
            x = x[:, :-1]
            if y[0].dim() > 0:
                y = y[:, 1:]
            else:
                warnings.warn(
                    "MidiTok DataCollator: You set shift_labels=True, but provided int"
                    "labels (for sequence classification tasks) which is suited for."
                    "Skipping label shifting.",
                    stacklevel=2,
                )

        # Add inputs / labels to output batch
        if x is not None:
            out_batch[self.inputs_kwarg_name] = x
        if y is not None:
            out_batch[self.labels_kwarg_name] = y

        # Create attention mask (just for padding, causal mask is handled by models)
        if x is not None:
            attention_mask = (x != self.pad_token).int()
            if attention_mask.dim() == 3:
                attention_mask = attention_mask[..., 0]  # (N,T,Z) --> (N,T)
            out_batch["attention_mask"] = attention_mask

        return out_batch


def _pad_batch(
    batch: Sequence[LongTensor],
    pad_token_id: int,
    pad_on_left: bool = False,
) -> LongTensor:
    r"""
    Pad sequences of a batch.

    :param batch: batch as a list of Tensors.
    :param pad_token_id: padding token id.
    :param pad_on_left: if given True, it will pad the sequences on the left. This can
        be required when using some libraries expecting padding on left, for example
        when generating with Hugging Face Transformers. (default: False)
    :return: the batch sequences, padded into a unique Tensor.
    """
    length_of_first = batch[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in batch)
    if are_tensors_same_length:
        return torch.stack(batch, dim=0).long()

    # Creating the full tensor and filling it with our data.
    if pad_on_left:
        return _pad_left(batch, pad_token_id)

    return torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=pad_token_id
    ).long()


def _pad_left(batch: Sequence[LongTensor], pad_token_id: int) -> LongTensor:
    r"""
    Pad sequences on the left, i.e. on the first indices.

    Padding on the left make the last element of each sequence the last token, which is
    convenient when generating autoregressively as a method can more easily and
    efficiently append the newly generated tokens.

    :param batch: batch as a list of Tensors.
    :param pad_token_id: padding token id.
    :return: the batch sequences, padded into a unique Tensor.
    """
    batch = [torch.flip(seq, dims=(0,)) for seq in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=pad_token_id
    )  # (N,T)
    return torch.flip(batch, dims=(1,)).long()
