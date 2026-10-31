"""A transparent, small-block Polar code encoder and SC decoder.

The implementation uses the Arikan transform G_N = F^{otimes log2(N)}
with F = [[1, 0], [1, 1]], over GF(2). It is intentionally written in
NumPy/Python so the encoding and decoding operations can be inspected.
"""
from __future__ import annotations

import numpy as np

N = 8
K = 4
INFO_POS = np.array([3, 5, 6, 7], dtype=int)
FROZEN_POS = np.array([0, 1, 2, 4], dtype=int)

_frozen_mask = np.zeros(N, dtype=bool)
_frozen_mask[FROZEN_POS] = True


def _validate_power_of_two(n: int) -> None:
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("Polar block length N must be a power of two >= 2.")


def polar_transform(u: np.ndarray) -> np.ndarray:
    """Apply x = u F^{otimes m} over GF(2), where len(u)=2^m."""
    x = np.asarray(u, dtype=np.uint8).copy()
    _validate_power_of_two(len(x))
    if np.any(x > 1):
        raise ValueError("Input vector must contain only binary values 0 and 1.")

    step = 1
    while step < len(x):
        for start in range(0, len(x), 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x


def polar_encode(info_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert K information bits into the selected positions and encode.

    Returns (u, x), where u is the N-bit vector before transformation and
    x is the N-bit Polar codeword.
    """
    bits = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
    if len(bits) != K:
        raise ValueError(f"Expected exactly K={K} information bits.")
    if np.any(bits > 1):
        raise ValueError("Information bits must be 0 or 1.")

    u = np.zeros(N, dtype=np.uint8)
    u[INFO_POS] = bits
    x = polar_transform(u)
    return u, x


def _f_llr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact box-plus update for LLRs: f(a,b)=2 atanh(tanh(a/2)tanh(b/2)).

    The logaddexp form is numerically stable and avoids overflow.
    Positive LLR means bit 0 is more likely.
    """
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def _g_llr(a: np.ndarray, b: np.ndarray, partial_sums: np.ndarray) -> np.ndarray:
    """SC g update. Cast partial sums to float to avoid uint8 underflow."""
    return b + (1.0 - 2.0 * partial_sums.astype(float)) * a


def sc_decode(channel_llr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Decode one N-bit LLR vector using recursive Successive Cancellation.

    Convention: positive channel LLR favours coded bit 0. The function
    returns the estimated full u vector and a trace of leaf decisions.
    """
    llr = np.asarray(channel_llr, dtype=float).reshape(-1)
    if len(llr) != N:
        raise ValueError(f"Expected exactly N={N} channel LLR values.")
    if not np.all(np.isfinite(llr)):
        raise ValueError("Channel LLR values must be finite.")

    trace: list[dict] = []

    def recurse(alpha: np.ndarray, frozen_mask: np.ndarray, offset: int):
        m = len(alpha)
        if m == 1:
            is_frozen = bool(frozen_mask[0])
            bit = 0 if is_frozen else int(alpha[0] < 0.0)
            trace.append({
                "position": offset,
                "llr": float(alpha[0]),
                "frozen": is_frozen,
                "decision": bit,
            })
            u_hat = np.array([bit], dtype=np.uint8)
            beta = u_hat.copy()
            return u_hat, beta

        half = m // 2
        a = alpha[:half]
        b = alpha[half:]

        # Decode the first half using the f update.
        alpha_left = _f_llr(a, b)
        u_left, beta_left = recurse(
            alpha_left, frozen_mask[:half], offset
        )

        # Decode the second half using the g update and left partial sums.
        alpha_right = _g_llr(a, b, beta_left)
        u_right, beta_right = recurse(
            alpha_right, frozen_mask[half:], offset + half
        )

        u_hat = np.concatenate((u_left, u_right))
        # Combine partial sums according to the polar transform.
        beta = np.concatenate((beta_left ^ beta_right, beta_right))
        return u_hat, beta

    u_hat, _ = recurse(llr, _frozen_mask, 0)
    return u_hat[INFO_POS], trace


def llr_from_bpsk(received: np.ndarray, noise_variance: float) -> np.ndarray:
    """Compute channel LLRs for BPSK mapping 0->+1, 1->-1."""
    if noise_variance <= 0:
        raise ValueError("Noise variance must be positive.")
    return 2.0 * np.asarray(received, dtype=float) / noise_variance
