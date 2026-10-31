"""End-to-end text transmission through Polar coding, BPSK and AWGN."""
from __future__ import annotations
import numpy as np

from channel import (
    awgn_channel,
    bpsk_modulate,
    ebn0_db_to_noise_variance,
)
from polar_code import INFO_POS, K, N, polar_encode, sc_decode, llr_from_bpsk
from text_codec import framed_bits_to_text, text_to_framed_bits


def transmit_message(message: str, ebn0_db: float, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    framed_bits, payload_bytes, padding_bits = text_to_framed_bits(message)
    noise_variance = ebn0_db_to_noise_variance(ebn0_db, K / N)
    sigma = float(np.sqrt(noise_variance))

    tx_blocks = []
    rx_info_blocks = []
    all_tx_info = []
    all_rx_info = []

    for block_index, start in enumerate(range(0, len(framed_bits), K)):
        information = framed_bits[start:start + K]
        u, coded = polar_encode(information)
        tx_symbols = bpsk_modulate(coded)
        received, noise = awgn_channel(tx_symbols, noise_variance, rng)
        channel_llr = llr_from_bpsk(received, noise_variance)
        decoded_info, trace = sc_decode(channel_llr)

        all_tx_info.extend(information.tolist())
        all_rx_info.extend(decoded_info.tolist())
        rx_info_blocks.extend(decoded_info.tolist())

        tx_blocks.append({
            "block": block_index + 1,
            "information_bits": information.copy(),
            "u_vector": u.copy(),
            "coded_bits": coded.copy(),
            "bpsk_symbols": tx_symbols.copy(),
            "noise": noise.copy(),
            "received": received.copy(),
            "channel_llr": channel_llr.copy(),
            "decoded_bits": decoded_info.copy(),
            "decoder_trace": trace,
        })

    decoded_framed = np.asarray(rx_info_blocks, dtype=np.uint8)
    header_valid = True
    try:
        recovered_message = framed_bits_to_text(decoded_framed)
    except (ValueError, UnicodeError):
        # At low SNR, even the length header can be corrupted. Report this
        # as a decoding failure instead of crashing the demonstration.
        recovered_message = "[Message could not be reconstructed: corrupted length header]"
        header_valid = False
    tx_arr = np.asarray(all_tx_info, dtype=np.uint8)
    rx_arr = np.asarray(all_rx_info, dtype=np.uint8)
    bit_errors = int(np.count_nonzero(tx_arr != rx_arr))
    ber = bit_errors / len(tx_arr) if len(tx_arr) else 0.0

    return {
        "original_message": message,
        "recovered_message": recovered_message,
        "snr_db": float(ebn0_db),
        "noise_variance": noise_variance,
        "noise_std": sigma,
        "payload_bytes": payload_bytes,
        "padding_bits": padding_bits,
        "framed_bits": framed_bits,
        "decoded_framed_bits": decoded_framed,
        "blocks": tx_blocks,
        "bit_errors": bit_errors,
        "total_bits": len(tx_arr),
        "ber": ber,
        "header_valid": header_valid,
        "success": header_valid and recovered_message == message,
    }
