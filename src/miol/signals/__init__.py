"""Answer-distribution signals along a chain of thought.

At checkpoints inside a cached completion (the prompt end, every blank-line boundary, the end), the model is
forced to answer ("... the final answer is \\boxed{") and its answer distribution is scored against exact
targets (the gold answer, its own final answer).  Linear probes are then fit on the cached hidden states
at those checkpoints.  See cluster/GUIDE_SIGNALS.md.
"""
