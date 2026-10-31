"""Monte-Carlo BER/FER simulation for Polar-coded BPSK over AWGN."""
from __future__ import annotations
import math
import numpy as np

from channel import awgn_channel, bpsk_modulate, ebn0_db_to_noise_variance
from polar_code import K, N, polar_encode, sc_decode, llr_from_bpsk


def simulate_ber(snr_values=range(16), frames_per_snr: int = 1500,
                 seed: int = 2026) -> dict:
    if frames_per_snr < 1:
        raise ValueError("frames_per_snr must be at least 1.")

    ber_values, fer_values = [], []
    bit_error_counts, frame_error_counts = [], []
    total_bits = frames_per_snr * K

    for snr_index, snr_db in enumerate(snr_values):
        rng = np.random.default_rng(seed + snr_index)
        noise_variance = ebn0_db_to_noise_variance(float(snr_db), K / N)
        bit_errors = 0
        frame_errors = 0

        for _ in range(frames_per_snr):
            info = rng.integers(0, 2, size=K, dtype=np.uint8)
            _, coded = polar_encode(info)
            symbols = bpsk_modulate(coded)
            received, _ = awgn_channel(symbols, noise_variance, rng)
            llr = llr_from_bpsk(received, noise_variance)
            decoded, _ = sc_decode(llr)

            errors = int(np.count_nonzero(info != decoded))
            bit_errors += errors
            frame_errors += int(errors > 0)

        ber_values.append(bit_errors / total_bits)
        fer_values.append(frame_errors / frames_per_snr)
        bit_error_counts.append(bit_errors)
        frame_error_counts.append(frame_errors)

    uncoded_ber = [
        0.5 * math.erfc(math.sqrt(10.0 ** (float(snr) / 10.0)))
        for snr in snr_values
    ]

    return {
        "snr_db": np.asarray(list(snr_values), dtype=float),
        "ber": np.asarray(ber_values, dtype=float),
        "fer": np.asarray(fer_values, dtype=float),
        "uncoded_ber": np.asarray(uncoded_ber, dtype=float),
        "bit_errors": np.asarray(bit_error_counts, dtype=int),
        "frame_errors": np.asarray(frame_error_counts, dtype=int),
        "frames_per_snr": frames_per_snr,
        "total_bits_per_snr": total_bits,
    }
