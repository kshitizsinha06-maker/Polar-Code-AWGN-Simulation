"""BPSK modulation and real AWGN channel helpers."""
from __future__ import annotations
import numpy as np


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if np.any(bits > 1):
        raise ValueError("Bits must contain only 0 and 1.")
    return 1.0 - 2.0 * bits.astype(float)


def ebn0_db_to_noise_variance(ebn0_db: float, code_rate: float) -> float:
    """Real AWGN variance for unit-energy BPSK and Eb/N0 (per info bit).

    sigma^2 = N0/2 = 1 / (2 * R * Eb/N0_linear)
    """
    if not 0.0 < code_rate <= 1.0:
        raise ValueError("Code rate must be in (0, 1].")
    ebn0_linear = 10.0 ** (float(ebn0_db) / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0_linear)


def awgn_channel(symbols: np.ndarray, noise_variance: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    symbols = np.asarray(symbols, dtype=float)
    noise = rng.normal(0.0, np.sqrt(noise_variance), size=symbols.shape)
    return symbols + noise, noise
