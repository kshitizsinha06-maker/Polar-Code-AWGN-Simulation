import numpy as np
from polar_code import K, N, INFO_POS, polar_encode, sc_decode
from channel import bpsk_modulate


def test_all_16_messages_decode_without_noise():
    for value in range(2**K):
        bits = np.array([(value >> shift) & 1 for shift in range(K-1, -1, -1)], dtype=np.uint8)
        _, codeword = polar_encode(bits)
        symbols = bpsk_modulate(codeword)
        # Very high-confidence channel LLRs, no noise.
        decoded, _ = sc_decode(1000.0 * symbols)
        assert np.array_equal(decoded, bits), (bits, codeword, decoded)


def test_encoder_returns_correct_lengths():
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    u, x = polar_encode(bits)
    assert len(u) == N
    assert len(x) == N
    assert np.array_equal(u[INFO_POS], bits)
    assert np.count_nonzero(u) <= K
