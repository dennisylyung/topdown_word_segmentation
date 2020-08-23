import math
from typing import List, Dict

import numpy as np
import torch
from transformers import AlbertModel, AlbertTokenizer, AlbertPreTrainedModel


class DataCollatorForCoherenceModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    def __init__(self, tokenizer: AlbertTokenizer, replace_prob=0.15, sequence_length=128, threshold=1e-5):
        super(DataCollatorForCoherenceModeling, self).__init__()
        self.tokenizer = tokenizer
        frequencies = []
        self.num_tokens = self.tokenizer.sp_model.GetPieceSize()
        for i in range(self.num_tokens):
            score = self.tokenizer.sp_model.GetScore(i)
            if (not score) or score == 0:
                frequencies.append(0.0)
            else:
                frequencies.append(math.exp(self.tokenizer.sp_model.GetScore(i)))
        frequencies = np.array(frequencies)
        #         sampling_weight = np.nan_to_num(frequencies * (np.sqrt(threshold / frequencies)))
        self.sampling_probs = frequencies / sum(frequencies)
        self.replace_prob = replace_prob
        self.sequence_length = sequence_length

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        input_j = self.replace_tokens(batch)
        return {"input_i": batch, "input_j": input_j, "labels": torch.ones(len(examples))}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            padded = torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            padded = torch.nn.utils.rnn.pad_sequence(examples, batch_first=True,
                                                     padding_value=self.tokenizer.pad_token_id)
        if padded.shape[1] < self.sequence_length:
            return torch.cat([padded,
                              torch.zeros((padded.shape[0], self.sequence_length - padded.shape[1]), dtype=torch.long)],
                             1)
        else:
            return padded

    def _get_random_tokens(self, shape):
        return np.random.choice(self.num_tokens, p=self.sampling_probs, size=shape)

    def replace_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        outputs = inputs.clone()

        # We sample a few tokens in each sequence for replacement (with probability args.replace_prob defaults to 0.15)
        probability_matrix = torch.full(inputs.shape, self.replace_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        replaced_indices = torch.bernoulli(probability_matrix).bool()

        # replace with random word
        indices_random = replaced_indices
        random_words = torch.LongTensor(self._get_random_tokens(inputs.shape))
        outputs[indices_random] = random_words[indices_random]

        return outputs


class AlbertCoherenceRank(AlbertPreTrainedModel):

    def __init__(self, config, sequence_length, mlp_sizes=None, mlp_dropout=0.5, freeze_albert=True):
        super().__init__(config)
        self.albert = AlbertModel(config)
        if freeze_albert:
            for p in self.albert.parameters():
                p.requires_grad = False

        mlp_layers = list()
        input_dim = self.albert.config.hidden_size * sequence_length
        if not mlp_sizes:
            mlp_sizes = [1024, 512, 256]
        for mlp_size in mlp_sizes:
            mlp_layers.append(torch.nn.Linear(input_dim, mlp_size))
            mlp_layers.append(torch.nn.LeakyReLU())
            mlp_layers.append(torch.nn.Dropout(p=mlp_dropout))
            input_dim = mlp_size
        mlp_layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*mlp_layers)

    def forward(self, input_i, input_j, target=None, **kwargs):
        """
        Calculate rank loss for two inputs, i and j. if taget is not provided, i should rank higher than j
        :param input_i: text 1. if target is not set, this should have higher coherence
        :param input_j: text 2. if target is not set, this should have lower coherence
        :param target: indicates ranking difference. 1 if i>j, -1 otherwise, defaults to 1
        :return: margin ranking loss, predicted i, predicted j
        """
        albert_out_i, _ = self.albert(input_i)
        predicted_i = self.mlp(torch.flatten(albert_out_i, start_dim=1))

        albert_out_j, _ = self.albert(input_j)
        predicted_j = self.mlp(torch.flatten(albert_out_j, start_dim=1))

        if not target:
            target = torch.ones(len(predicted_i)).to(self.device)

        loss = - ((predicted_i - predicted_j) * target).sigmoid().log().mean()

        return loss, predicted_i, predicted_j
