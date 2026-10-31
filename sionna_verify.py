"""Optional cross-check against NVIDIA Sionna's Polar encoder and SC decoder.

Install Sionna separately (see README.md) before running this file.
"""
from __future__ import annotations
import numpy as np
import torch

from sionna.phy.fec.polar import PolarEncoder, PolarSCDecoder
from polar_code import FROZEN_POS, INFO_POS, K, N, polar_encode, sc_decode


def main() -> None:
    reference_encoder = PolarEncoder(FROZEN_POS, N, device="cpu")
    reference_decoder = PolarSCDecoder(FROZEN_POS, N, device="cpu")

    checked = 0
    for value in range(2**K):
        info = np.array(
            [(value >> shift) & 1 for shift in range(K - 1, -1, -1)],
            dtype=np.uint8,
        )
        _, our_codeword = polar_encode(info)

        torch_info = torch.tensor(info[None, :], dtype=torch.float32)
        sionna_codeword = reference_encoder(torch_info).detach().cpu().numpy()[0]
        sionna_codeword = np.rint(sionna_codeword).astype(np.uint8)
        assert np.array_equal(our_codeword, sionna_codeword), (
            "Encoder mismatch", info, our_codeword, sionna_codeword
        )

        # Our channel LLR is positive for bit 0; Sionna's documented example
        # uses logits positive for bit 1, so invert the sign for its decoder.
        our_channel_llr = 30.0 * (1.0 - 2.0 * our_codeword.astype(float))
        our_decoded, _ = sc_decode(our_channel_llr)
        sionna_logits = torch.tensor(-our_channel_llr[None, :], dtype=torch.float32)
        sionna_decoded = reference_decoder(sionna_logits).detach().cpu().numpy()[0]
        sionna_decoded = np.rint(sionna_decoded).astype(np.uint8)

        assert np.array_equal(our_decoded, info)
        assert np.array_equal(sionna_decoded, info), (
            "Decoder mismatch", info, our_decoded, sionna_decoded
        )
        checked += 1

    print(f"PASS: custom encoder/SC decoder match Sionna for {checked} noiseless codewords.")


if __name__ == "__main__":
    main()
