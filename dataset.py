import logging
import os

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BatchedLineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        encoded = self.tokenizer.encode(self.examples[i],
                                        add_special_tokens=True,
                                        truncation=True,
                                        max_length=self.block_size)
        return torch.tensor(encoded, dtype=torch.long)
