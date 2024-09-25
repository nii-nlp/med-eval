from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch


@dataclass
class LMDataCollatorForPerplexity:
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        input_ids, labels = tuple([b[key] for b in batch] for key in ("input_ids", "labels"))

        input_ids = [torch.tensor(i) for i in input_ids]
        labels = [torch.tensor(i) for i in labels]

        # Padding on the right  (计算PPL用这个)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            request_id=[b["request_id"] for b in batch],
            option_id=[b["option_id"] for b in batch],
            sample=[b["sample"] for b in batch]
        )


@dataclass
class LMDataCollatorForGeneration:
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]

        # Padding on the left (生成用这个)
        input_ids = [torch.tensor(i[::-1]) for i in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,  # reverse the list and create tensors
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).flip(dims=[1])  # reverse/flip the padded tensor in first dimension

        return dict(
            input_ids=input_ids,
            request_id=[b["request_id"] for b in batch],
            sample=[b["sample"] for b in batch]
        )
