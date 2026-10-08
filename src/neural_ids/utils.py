"""Small helpers shared by the whole project.

`set_seed` is the single entry point for randomness (see CLAUDE.md:
"Randomness only through `neural_ids.utils.set_seed` or an explicit `np.random.Generator`").

Why randomness needs control in machine learning:
- Neural networks start from random weights, and training shuffles the data in random order.
- Without a fixed seed, every run gives slightly different numbers, so a result cannot be
  checked or reproduced. With a fixed seed, the same code gives the same numbers every time.
"""

# `hashlib` (standard library) computes cryptographic hashes such as SHA-256.
import hashlib

# `random` is Python's built-in (standard library) random number module.
import random
from pathlib import Path

# NumPy is the array library we use for all numeric work. `np` is the usual short name.
import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python's `random` module and return a NumPy random Generator.

    A *seed* is the starting value of a pseudo-random number generator (PRNG). A PRNG is a
    deterministic algorithm: the same seed always produces the same sequence of numbers.

    Why return a Generator instead of calling the global `np.random.seed(seed)`?
    - The global seed is shared state: any library that also calls `np.random` changes the
      sequence behind our back, so results depend on call order in unrelated code.
    - A Generator is a normal object (like `java.util.Random` in Java). We pass it as an
      argument to the functions that need randomness, so each function's randomness is
      explicit, local, and reproducible.

    Args:
        seed: Any non-negative integer, for example 42.

    Returns:
        A `np.random.Generator` created from `seed`. Two calls with the same seed return
        Generators that produce identical numbers.

    Example:
        >>> rng = set_seed(42)
        >>> x = rng.normal(size=(3, 2))  # x: (3, 2) array of standard normal numbers

    Complexity: O(1) time and O(1) memory (the generator state has a fixed size).
    """
    # 1. Seed Python's built-in `random` module. This covers any code that uses `random`
    #    (for example `random.shuffle`), so it also becomes reproducible.
    random.seed(seed)
    # 2. Create a new, independent NumPy Generator from the seed and return it.
    #    `default_rng` uses the PCG64 algorithm, which is fast and statistically strong.
    #    Callers keep this object and pass it to functions that need random numbers.
    return np.random.default_rng(seed)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, as 64 lowercase hex characters.

    Used as a fingerprint: two files with the same hash have the same content (a collision is
    practically impossible), so it links derived files back to the exact table they came from.
    Reads in 1 MiB chunks: O(file size) time, O(1) memory.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
