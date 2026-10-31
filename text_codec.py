"""Convert UTF-8 text to/from a stream of bits with a 16-bit length header."""
from __future__ import annotations
import numpy as np
from polar_code import K

HEADER_BITS = 16


def text_to_framed_bits(message: str) -> tuple[np.ndarray, int, int]:
    payload = message.encode("utf-8")
    if len(payload) > 65535:
        raise ValueError("Message is too long; maximum is 65,535 UTF-8 bytes.")

    length_header = np.unpackbits(
        np.array([len(payload) >> 8, len(payload) & 0xFF], dtype=np.uint8)
    )
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    framed = np.concatenate((length_header, payload_bits)).astype(np.uint8)
    padding = (-len(framed)) % K
    if padding:
        framed = np.concatenate((framed, np.zeros(padding, dtype=np.uint8)))
    return framed, len(payload), padding


def framed_bits_to_text(bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) < HEADER_BITS:
        raise ValueError("Decoded bitstream is too short to contain the length header.")

    header_bytes = np.packbits(bits[:HEADER_BITS])
    payload_length = int(header_bytes[0]) * 256 + int(header_bytes[1])
    payload_end = HEADER_BITS + payload_length * 8
    if payload_end > len(bits):
        raise ValueError("Decoded frame is incomplete; length header exceeds received data.")

    payload = np.packbits(bits[HEADER_BITS:payload_end]).tobytes()
    return payload.decode("utf-8", errors="replace")
