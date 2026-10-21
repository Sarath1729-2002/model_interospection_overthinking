"""
Infinite sampler with replacement for continuous training without epochs.
"""

import torch
from torch.utils.data import Sampler


class InfiniteSamplerWithReplacement(Sampler):
    """
    Sampler that yields infinite random samples with replacement.

    This allows training to continue indefinitely without epoch boundaries,
    with each batch seeing random samples from the dataset.

    Args:
        data_source: Dataset to sample from
        shuffle: Whether to shuffle (always True for this sampler)
        seed: Random seed for reproducibility
    """

    def __init__(self, data_source, shuffle=True, seed=None):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = len(data_source)

        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

    def __iter__(self):
        """Yield infinite random indices with replacement."""
        while True:
            # Generate random indices with replacement
            if self.generator is not None:
                indices = torch.randint(
                    high=self.num_samples,
                    size=(self.num_samples,),
                    generator=self.generator,
                    dtype=torch.int64
                )
            else:
                indices = torch.randint(
                    high=self.num_samples,
                    size=(self.num_samples,),
                    dtype=torch.int64
                )

            yield from indices.tolist()

    def __len__(self):
        """
        Return a large number to avoid issues with progress bars.
        This is never actually used since the sampler is infinite.
        """
        return 10**9  # Arbitrarily large number
