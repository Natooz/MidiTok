"""
Collator objects for PyTorch `DataLoader`s.
"""
from typing import List, Any, Dict
from copy import deepcopy

from torch import LongTensor
import torch


class DataCollator:
    def __init__(
        self,
        pad_token_id: int,
        bos_token_id: int = None,
        eos_token_id: int = None,
        pad_on_left: bool = False,
        copy_inputs_as_labels: bool = False,
        shift_labels: bool = False,
        labels_pad_idx: int = -100,
        inputs_kwarg_name: str = "input_ids",
        labels_kwarg_name: str = "labels",
    ):
        r"""Multifunction data collator, applying padding (right or left), allowing to add `BOS` and `EOS` tokens.
        It will also add an "attention_mask" entry to the batch, following the padding applied.

        :param pad_token_id: padding token id.
        :param bos_token_id: BOS token id. (default: None)
        :param eos_token_id: EOS token id. (default: None)
        :param pad_on_left: if given True, it will pad the sequences on the left. This can be required when using
            some libraries expecting padding on left, for example when generating with Hugging Face Transformers.
            (default: False)
        :param copy_inputs_as_labels: will add a labels entry (`inputs_kwarg_name`) to the batch
            (or replace the existing one), which is a copy to the input entry (`labels_kwarg_name`). (default: False)
        :param shift_labels: will shift inputs and labels for autoregressive training / teacher forcing.
            (default: False)
        :param labels_pad_idx: padding id for labels. (default: -100)
        :param inputs_kwarg_name: name of dict / kwarg key for inputs. (default: "input_ids")
        :param labels_kwarg_name: name of dict / kwarg key for inputs. (default: "labels")
        """
        self.pad_token = pad_token_id
        self.bos_token = bos_token_id
        self.eos_token = eos_token_id
        self.pad_on_left = pad_on_left
        self.copy_inputs_as_labels = copy_inputs_as_labels
        self.shift_labels = shift_labels
        self.labels_pad_idx = labels_pad_idx
        self.inputs_kwarg_name = inputs_kwarg_name
        self.labels_kwarg_name = labels_kwarg_name

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, LongTensor]:
        out_batch = {}
        x, y = None, None

        # Figure out inputs + adds BOS and EOS tokens
        if self.inputs_kwarg_name in batch[0]:
            x = [seq[self.inputs_kwarg_name] for seq in batch]
            _add_bos_eos_tokens_to_batch(
                x,
                bos_tok_id=self.bos_token,
                eos_tok_id=self.eos_token,
            )

        # Figure out labels + adds BOS and EOS tokens
        if self.labels_kwarg_name in batch[0]:
            y = [seq[self.labels_kwarg_name] for seq in batch]
            _add_bos_eos_tokens_to_batch(
                y,
                bos_tok_id=self.bos_token,
                eos_tok_id=self.eos_token,
            )
        elif self.copy_inputs_as_labels:
            y = deepcopy(x)

        # Pad inputs / convert to Tensors
        if x is not None:
            x = _pad_batch(x, self.pad_token, self.pad_on_left)
        if y is not None:
            if isinstance(y[0], LongTensor):
                y = _pad_batch(y, self.labels_pad_idx, self.pad_on_left)
            else:  # classification
                y = torch.stack(y)

        # Shift labels, otherwise it's handled by models
        if self.shift_labels:
            x = x[:, :-1]
            y = y[:, 1:]

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


def _add_bos_eos_tokens_to_batch(
    batch: List[LongTensor],
    bos_tok_id: int = None,
    eos_tok_id: int = None,
):
    """Adds (inplace) BOS and EOS tokens to inputs.

    :param batch: batch as a list of Tensors.
    :param bos_tok_id: BOS token id. (default: None)
    :param eos_tok_id: EOS token id. (default: None)
    """
    if bos_tok_id is None and eos_tok_id is None:
        return

    sos_shape = list(batch[0].shape)
    sos_shape[0] = 1  # (1) or (1,Z)
    for i in range(len(batch)):
        if bos_tok_id is not None and eos_tok_id is not None:
            batch[i] = torch.cat(
                [
                    torch.full(sos_shape, bos_tok_id),
                    batch[i],
                    torch.full(sos_shape, eos_tok_id),
                ],
                dim=0,
            ).long()
        elif bos_tok_id is not None:
            batch[i] = torch.cat(
                [torch.full(sos_shape, bos_tok_id), batch[i]], dim=0
            ).long()
        else:  # EOS not None
            batch[i] = torch.cat(
                [batch[i], torch.full(sos_shape, eos_tok_id)], dim=0
            ).long()


def _pad_batch(
    batch: List[LongTensor],
    pad_token_id: int,
    pad_on_left: bool = False,
) -> LongTensor:
    r"""Pad sequences of a batch.

    :param batch: batch as a list of Tensors.
    :param pad_token_id: padding token id.
    :param pad_on_left: if given True, it will pad the sequences on the left. This can be required when using
        some libraries expecting padding on left, for example when generating with Hugging Face Transformers.
        (default: False)
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
    else:
        return torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=pad_token_id
        ).long()


def _pad_left(batch: List[LongTensor], pad_token_id: int) -> LongTensor:
    r"""Here the sequences are padded to the left, so that the last token along the time dimension
    is always the last token of each seq, allowing to efficiently generate by batch

    :param batch: batch as a list of Tensors.
    :param pad_token_id: padding token id.
    :return: the batch sequences, padded into a unique Tensor.
    """
    batch = [torch.flip(seq, dims=(0,)) for seq in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=pad_token_id
    )  # (N,T)
    batch = torch.flip(batch, dims=(1,)).long()
    return batch
