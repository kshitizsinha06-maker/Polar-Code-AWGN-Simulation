# Polar Encoding and Decoding over an AWGN Channel

An educational end-to-end simulation of text transmission using a small Polar code, BPSK modulation, a real AWGN channel, and a recursive Successive Cancellation (SC) decoder.

## Features

- Enter a UTF-8 text message at the transmitter.
- Convert the message to bits and prepend a 16-bit payload-length header.
- Split the bitstream into 4-bit information blocks.
- Insert frozen bits and apply the Polar transform (`N=8`, `K=4`).
- BPSK modulate (`0 -> +1`, `1 -> -1`).
- Add AWGN with a selectable `Eb/N0` from 0 to 15 dB.
- Compute channel LLRs and recursively SC-decode each codeword.
- Reconstruct the text at the receiver and compare it with the original.
- Run a Monte-Carlo SNR sweep and plot BER/FER.

## Code parameters

| Parameter | Value |
|---|---:|
| Block length N | 8 |
| Information bits K | 4 |
| Code rate R | 1/2 |
| Information positions | 3, 5, 6, 7 |
| Frozen positions | 0, 1, 2, 4 |
| Modulation | BPSK |
| Channel | Real AWGN |
| Decoder | Recursive SC |
| SNR range | 0 to 15 dB (`Eb/N0`) |

The construction is intentionally small and educational. It is not a 5G NR Polar code construction.

## Setup (Windows / VS Code)

Open a terminal in this directory and run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is unavailable, use `python -m venv .venv` with your installed Python version.

## Run the application

```powershell
streamlit run app.py
```

Streamlit will print a local URL (normally `http://localhost:8501`). Open it in a browser.

## Run a quick core test without Streamlit

```powershell
python -c "from transmission import transmit_message; r=transmit_message('HELLO', 15, 7); print(r['recovered_message'], r['ber'])"
```

## Model details

The encoder applies the Arikan transform `G_N = F^(otimes log2(N))` over GF(2), with `F = [[1,0],[1,1]]`.

BPSK uses unit-energy symbols, `0 -> +1` and `1 -> -1`. For real AWGN and `Eb/N0` per information bit:

`noise_variance = 1 / (2 * R * 10**(EbN0_dB/10))`.

The channel LLR is `2*y/noise_variance`, so a positive LLR favours bit 0. The recursive SC decoder uses exact box-plus `f` updates and `g` updates with partial sums.

The 16-bit length header is transmitted as part of the bitstream so the receiver can remove padding and recover the exact UTF-8 payload length. The demo assumes the transmitter and receiver agree on this framing convention.

## Optional Sionna cross-check

NVIDIA Sionna provides native `PolarEncoder` and `PolarSCDecoder` blocks. The assignment app does not need Sionna, so install it only if you have time and want to verify the custom implementation against a library reference.

```powershell
pip install -r requirements-sionna.txt
python sionna_verify.py
```

The reference script checks all 16 possible 4-bit messages. It also accounts for the LLR sign convention: this project uses positive LLR for bit 0, while Sionna's documented example uses logits positive for bit 1. Sionna PHY requires Python 3.11+ and PyTorch 2.9+ in the current release line.

- https://nvlabs.github.io/sionna/v2.1.0/phy/api/fec/polar/index.html
- https://nvlabs.github.io/sionna/v2.1.0/phy/api/fec/polar/sionna.phy.fec.polar.PolarEncoder.html
- https://nvlabs.github.io/sionna/v2.1.0/phy/api/fec/polar/sionna.phy.fec.polar.PolarSCDecoder.html

## Important interpretation note

A measured BER/FER of zero means no errors occurred in the finite simulation run; it does not mean the true error probability is exactly zero. The BER plot displays zero-count points at the one-error measurement floor.

## Made by KSHITIZ SINHA, 2K24 BIT Mesra