"""Streamlit UI for Polar coding over an AWGN channel."""
from __future__ import annotations

import html
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from performance import simulate_ber
from polar_code import INFO_POS, FROZEN_POS, K, N
from transmission import transmit_message


# -----------------------------------------------------------------------------
# Page configuration and visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Polar Coding over AWGN",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --bg: #080d19;
      --panel: #101a2d;
      --panel-2: #14213a;
      --line: rgba(148, 173, 213, .16);
      --muted: #91a2bf;
      --text: #edf4ff;
      --cyan: #67e8f9;
      --violet: #a78bfa;
      --green: #6ee7b7;
      --amber: #fbbf24;
    }

    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp {
      background:
        radial-gradient(900px 460px at 5% -10%, rgba(34, 211, 238, .12), transparent 62%),
        radial-gradient(760px 440px at 100% 0%, rgba(139, 92, 246, .12), transparent 58%),
        var(--bg);
      color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(8, 13, 25, .78); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { padding-top: 2.1rem; padding-bottom: 3.2rem; max-width: 1440px; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1628 0%, #0a1120 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c5d2e8; }

    h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -.035em; }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 750 !important; }
    p, label, li { color: #c5d2e8; }
    small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

    .hero {
      position: relative; overflow: hidden;
      padding: 2rem 2.1rem 1.8rem;
      border: 1px solid rgba(122, 163, 224, .24);
      border-radius: 24px;
      background: linear-gradient(118deg, rgba(18, 36, 62, .96), rgba(24, 26, 55, .92) 58%, rgba(35, 27, 68, .94));
      box-shadow: 0 24px 70px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.06);
      margin-bottom: 1.35rem;
    }
    .hero:after {
      content: ''; position: absolute; width: 320px; height: 320px; right: -75px; top: -170px;
      border-radius: 50%; border: 1px solid rgba(103,232,249,.25);
      box-shadow: 0 0 0 28px rgba(103,232,249,.035), 0 0 0 58px rgba(167,139,250,.035);
    }
    .hero-kicker { color: var(--cyan); font: 500 11px 'DM Mono', monospace; letter-spacing: .19em; text-transform: uppercase; }
    .hero-title { font-size: clamp(2rem, 4vw, 3.15rem); line-height: 1.06; margin: .45rem 0 .7rem; color: #f5f8ff; font-weight: 800; letter-spacing: -.055em; }
    .hero-title span { background: linear-gradient(90deg, #67e8f9, #a78bfa); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .hero-subtitle { max-width: 760px; color: #b4c4dd; font-size: .98rem; line-height: 1.65; margin-bottom: 1.15rem; }
    .hero-chips { display: flex; flex-wrap: wrap; gap: .5rem; }
    .hero-chip { border: 1px solid rgba(148, 190, 231, .22); background: rgba(8, 16, 31, .42); padding: .35rem .65rem; border-radius: 999px; color: #d9e9ff; font: 11px 'DM Mono', monospace; }

    .brand-lockup { display: flex; align-items: center; gap: .72rem; margin: .1rem 0 1.35rem; }
    .brand-icon { width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center; font-size: 21px; color: #06121d; background: linear-gradient(135deg, #67e8f9, #a78bfa); box-shadow: 0 8px 24px rgba(103,232,249,.18); }
    .brand-name { color: #f4f8ff; font-size: 1.05rem; font-weight: 800; letter-spacing: -.04em; }
    .brand-sub { color: #8297b7; font: 10px 'DM Mono', monospace; letter-spacing: .13em; text-transform: uppercase; margin-top: 2px; }

    .section-head { display: flex; align-items: center; gap: .8rem; margin: .2rem 0 .9rem; }
    .section-index { width: 32px; height: 32px; flex: 0 0 32px; display: grid; place-items: center; border-radius: 10px; color: var(--cyan); background: rgba(103,232,249,.1); border: 1px solid rgba(103,232,249,.22); font: 500 12px 'DM Mono', monospace; }
    .section-title { color: #edf4ff; font-weight: 800; font-size: 1.12rem; letter-spacing: -.025em; }
    .section-note { color: #91a2bf; font-size: .82rem; margin-top: 2px; }

    .mini-label { color: #7f95b7; font: 10px 'DM Mono', monospace; letter-spacing: .15em; text-transform: uppercase; }
    .metric-tile { min-height: 98px; padding: 1rem 1.05rem; border-radius: 16px; border: 1px solid var(--line); background: linear-gradient(145deg, rgba(20,33,58,.94), rgba(13,23,41,.94)); box-shadow: 0 10px 30px rgba(0,0,0,.12); }
    .metric-value { color: #f2f7ff; font-size: 1.55rem; font-weight: 800; letter-spacing: -.045em; margin-top: .4rem; }
    .metric-foot { color: #8ea2c2; font-size: .72rem; margin-top: .15rem; }
    .status-pill { display: inline-block; border-radius: 999px; padding: .28rem .62rem; font: 10px 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .status-good { color: #8cf0c7; border: 1px solid rgba(110,231,183,.3); background: rgba(16,185,129,.1); }
    .status-warn { color: #ffd782; border: 1px solid rgba(251,191,36,.3); background: rgba(245,158,11,.1); }
    .muted-box { border: 1px solid var(--line); background: rgba(17, 29, 50, .72); border-radius: 16px; padding: 1rem 1.1rem; }
    .pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: .55rem; margin: .5rem 0 1.3rem; }
    .pipeline-step { border: 1px solid var(--line); background: rgba(17,29,50,.72); padding: .72rem .8rem; border-radius: 13px; }
    .pipeline-step b { color: #e4efff; display: block; font-size: .78rem; }
    .pipeline-step span { color: #8196b7; font: 10px 'DM Mono', monospace; }
    .pipeline-step.active { border-color: rgba(103,232,249,.35); background: linear-gradient(130deg, rgba(34,211,238,.12), rgba(139,92,246,.09)); }

    div[data-testid="stTabs"] > div[role="tablist"] { gap: .45rem; border-bottom: 1px solid var(--line); }
    button[data-baseweb="tab"] { border-radius: 11px 11px 0 0; padding: .75rem 1rem; color: #93a5c1; font-weight: 700; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #eafaff; background: rgba(103,232,249,.08); border-bottom: 2px solid var(--cyan); }
    div[data-testid="stButton"] > button[kind="primary"] { border: 0; border-radius: 12px; color: #07131f; font-weight: 800; background: linear-gradient(100deg, #67e8f9, #a5b4fc); box-shadow: 0 8px 24px rgba(103,232,249,.16); transition: transform .15s ease, box-shadow .15s ease; }
    div[data-testid="stButton"] > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 12px 30px rgba(103,232,249,.22); }
    div[data-testid="stButton"] > button { border-radius: 12px; }
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input { background: #0c1729; color: #eff6ff; border-color: rgba(148,173,213,.2); border-radius: 11px; }
    div[data-testid="stSelectbox"] > div, div[data-testid="stSlider"] { color: #eaf1ff; }
    div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 13px; background: rgba(13,23,41,.58); }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(20,33,58,.94), rgba(13,23,41,.94)); border: 1px solid var(--line); padding: .9rem 1rem; border-radius: 15px; }
    div[data-testid="stMetricLabel"] { color: #8fa4c5 !important; }
    div[data-testid="stMetricValue"] { color: #f1f6ff !important; }
    div[data-testid="stAlert"] { border-radius: 13px; }
    hr { border-color: var(--line); }
    .footer { text-align: center; color: #7185a5; font: 10px 'DM Mono', monospace; letter-spacing: .1em; padding-top: 1.5rem; }

    @media (max-width: 800px) {
      .hero { padding: 1.35rem; }
      .pipeline { grid-template-columns: repeat(2, 1fr); }
      .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_head(index: str, title: str, note: str = "") -> None:
    st.markdown(
        f"""<div class="section-head">
          <div class="section-index">{index}</div>
          <div><div class="section-title">{title}</div>
          <div class="section-note">{note}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, foot: str) -> None:
    st.markdown(
        f"""<div class="metric-tile"><div class="mini-label">{label}</div>
        <div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>""",
        unsafe_allow_html=True,
    )


def style_axis(ax) -> None:
    ax.set_facecolor("#101a2d")
    ax.figure.set_facecolor("#101a2d")
    ax.tick_params(colors="#9db0ce", labelsize=9)
    ax.xaxis.label.set_color("#c8d6eb")
    ax.yaxis.label.set_color("#c8d6eb")
    ax.title.set_color("#edf4ff")
    for spine in ax.spines.values():
        spine.set_color("#2a3a57")
    ax.grid(True, which="both", color="#7185a5", alpha=.18, linestyle="--")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="mini-label">SYSTEM CONFIGURATION</div>', unsafe_allow_html=True)
    st.markdown("---")
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Block length", f"N = {N}")
    with s2:
        st.metric("Payload / block", f"K = {K}")
    st.metric("Code rate", f"R = {K/N:.2f}")
    st.markdown(
        f"""<div class="muted-box">
        <div class="mini-label">PHY LAYER</div>
        <p style="margin:.55rem 0 .25rem;color:#e5efff"><b>Modulation</b> · BPSK</p>
        <p style="margin:.25rem 0;color:#e5efff"><b>Channel</b> · Real AWGN</p>
        <p style="margin:.25rem 0;color:#e5efff"><b>Decoder</b> · Recursive SC</p>
        <p style="margin:.25rem 0 0;color:#e5efff"><b>SNR sweep</b> · 0 to 15 dB</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(" ")
    with st.expander("Code construction", expanded=False):
        st.write(f"**Information positions:** `{INFO_POS.tolist()}`")
        st.write(f"**Frozen positions:** `{FROZEN_POS.tolist()}`")
        st.caption("Educational N=8, K=4 construction; not a 5G NR construction.")
    st.markdown("---")
    st.caption("POLAR CODING  /  BPSK  /  AWGN  /  SC DECODING")


# -----------------------------------------------------------------------------
# Hero and overview
# -----------------------------------------------------------------------------
st.markdown(
    """<div class="hero">
      <div class="hero-kicker">DIGITAL COMMUNICATIONS · INTERACTIVE LAB</div>
      <div class="hero-title">Polar <span>Code</span></div>
      <div class="hero-subtitle">Explore how Polar codes protect a message as it travels through a noisy channel. Inspect every stage from information bits to the recovered message, then measure BER and FER across the SNR range.</div>
      <div class="hero-chips"><span class="hero-chip">N = 8 / K = 4</span><span class="hero-chip">RATE 1/2</span><span class="hero-chip">BPSK</span><span class="hero-chip">AWGN CHANNEL</span><span class="hero-chip">SUCCESSIVE CANCELLATION</span></div>
    </div>""",
    unsafe_allow_html=True,
)

ov1, ov2, ov3, ov4 = st.columns(4)
with ov1:
    metric_tile("CODEWORD LENGTH", "08 bits", "Each encoded block")
with ov2:
    metric_tile("INFORMATION", "04 bits", "Per Polar block")
with ov3:
    metric_tile("CODE RATE", "0.50", "K / N")
with ov4:
    metric_tile("CHANNEL", "AWGN", "Real-valued Gaussian noise")

st.markdown(" ")
message_tab, performance_tab, theory_tab = st.tabs(
    ["✉  Message studio", "⌁  Performance lab", "∑  Theory & model"]
)


# -----------------------------------------------------------------------------
# Message transmission studio
# -----------------------------------------------------------------------------
with message_tab:
    section_head("01", "Message transmission", "Compose a payload, choose channel conditions, and inspect the receiver output.")
    st.markdown(
        """<div class="pipeline">
          <div class="pipeline-step active"><span>01 / SOURCE</span><b>Text → bits</b></div>
          <div class="pipeline-step"><span>02 / ENCODER</span><b>Polar + BPSK</b></div>
          <div class="pipeline-step"><span>03 / CHANNEL</span><b>AWGN noise</b></div>
          <div class="pipeline-step"><span>04 / RECEIVER</span><b>SC → text</b></div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        message_col, settings_col = st.columns([1.65, 1], gap="large")
        with message_col:
            st.markdown('<div class="mini-label">TRANSMITTER INPUT</div>', unsafe_allow_html=True)
            message = st.text_area(
                "Message payload",
                value="HELLO WORLD",
                height=112,
                help="Enter any UTF-8 text. The application frames it and encodes it block by block.",
                placeholder="Type a message to transmit…",
                label_visibility="collapsed",
            )
            msg_bytes = len(message.encode("utf-8"))
            st.caption(f"{msg_bytes} payload bytes  ·  {msg_bytes * 8} payload bits  ·  UTF-8")
        with settings_col:
            st.markdown('<div class="mini-label">CHANNEL CONTROLS</div>', unsafe_allow_html=True)
            snr_db = st.slider("Eb/N0 · signal-to-noise ratio", min_value=0, max_value=15, value=8, step=1)
            seed = st.number_input("Noise seed", min_value=0, max_value=999999, value=7, step=1)
            st.caption("A fixed seed makes the noise realization repeatable.")
            st.markdown(
                f'<span class="status-pill status-good">{snr_db} dB selected</span>',
                unsafe_allow_html=True,
            )
        run_demo = st.button("▶  Transmit and decode message", type="primary", use_container_width=True)

    if run_demo:
        try:
            with st.spinner("Encoding blocks · applying AWGN · decoding at receiver…"):
                st.session_state["transmission_result"] = transmit_message(message, snr_db, int(seed))
        except Exception as exc:
            st.error(f"Transmission failed: {exc}")

    result = st.session_state.get("transmission_result")
    if result is not None:
        st.markdown(" ")
        section_head("02", "Transmission trace", "Inspect how the original text is framed and transformed into Polar codewords.")
        tx1, tx2, tx3 = st.columns(3)
        with tx1:
            metric_tile("PAYLOAD SIZE", f"{result['payload_bytes']} B", "UTF-8 bytes")
        with tx2:
            metric_tile("FRAMED STREAM", f"{len(result['framed_bits'])} bits", f"Padding: {result['padding_bits']} bits")
        with tx3:
            metric_tile("POLAR BLOCKS", str(len(result["blocks"])), "4 information → 8 coded bits")

        with st.expander("01 · Source encoding and framing", expanded=True):
            st.markdown("**Original message**")
            st.code(result["original_message"] or "(empty message)", language="text")
            st.caption("The UTF-8 payload is preceded by a 16-bit length header so the receiver can remove padding.")
            st.markdown("**UTF-8 payload bits**")
            st.code(" ".join(f"{b:08b}" for b in result["original_message"].encode("utf-8")) or "(empty message)", language="text")
            st.markdown("**Framed bitstream**")
            st.code(" ".join(map(str, result["framed_bits"].tolist())), language="text")

        with st.expander("02 · Polar encoding, BPSK modulation and AWGN channel", expanded=True):
            a, b, c = st.columns(3)
            with a:
                st.metric("Eb/N0", f"{result['snr_db']:.1f} dB")
            with b:
                st.metric("Noise variance", f"{result['noise_variance']:.5f}")
            with c:
                st.metric("Noise σ", f"{result['noise_std']:.5f}")
            st.caption("Information bits are inserted at positions [3, 5, 6, 7]. Frozen positions [0, 1, 2, 4] are set to zero before applying the Polar transform.")
            rows = []
            for block in result["blocks"]:
                rows.append({
                    "Block": block["block"],
                    "Information bits": "".join(map(str, block["information_bits"].tolist())),
                    "Input u": "".join(map(str, block["u_vector"].tolist())),
                    "Encoded x": "".join(map(str, block["coded_bits"].tolist())),
                    "Decoded bits": "".join(map(str, block["decoded_bits"].tolist())),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("03 · Deep inspection: channel samples and SC decisions", expanded=False):
            for block in result["blocks"]:
                with st.expander(f"Block {block['block']:02d} · inspect vectors", expanded=False):
                    detail = {
                        "Information bits": block["information_bits"],
                        "Input vector u": block["u_vector"],
                        "Polar codeword x": block["coded_bits"],
                        "BPSK symbols": np.round(block["bpsk_symbols"], 4),
                        "AWGN noise samples": np.round(block["noise"], 4),
                        "Received samples y": np.round(block["received"], 4),
                        "Channel LLRs": np.round(block["channel_llr"], 4),
                        "Recovered information": block["decoded_bits"],
                    }
                    st.json({key: np.asarray(value).tolist() for key, value in detail.items()})
                    trace_rows = [{
                        "u position": item["position"],
                        "Frozen?": "Yes" if item["frozen"] else "No",
                        "Bit-channel LLR": round(item["llr"], 4),
                        "SC decision": item["decision"],
                    } for item in block["decoder_trace"]]
                    st.markdown("**SC decoder leaf decisions**")
                    st.dataframe(trace_rows, use_container_width=True, hide_index=True)

        st.markdown(" ")
        section_head("03", "Receiver result", "Compare the original payload with the reconstructed UTF-8 message.")
        status_col, recovered_col = st.columns([1, 1.35], gap="large")
        with status_col:
            st.metric("Information-bit errors", f"{result['bit_errors']} / {result['total_bits']}")
            st.metric("Message BER", f"{result['ber']:.6f}")
            if result["success"]:
                st.markdown('<span class="status-pill status-good">● MESSAGE RECOVERED</span>', unsafe_allow_html=True)
                st.success("The receiver reconstructed the original message exactly.")
            else:
                st.markdown('<span class="status-pill status-warn">● DECODE MISMATCH</span>', unsafe_allow_html=True)
                st.warning("The receiver could not reconstruct the exact message. Noise may have corrupted information bits or the length header.")
        with recovered_col:
            st.markdown('<div class="mini-label">RECEIVER OUTPUT</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="muted-box" style="min-height:150px"><div style="color:#8fa4c5;font-size:11px;margin-bottom:10px">DECODED PAYLOAD</div><div style="font-size:1.25rem;font-weight:750;color:#f0f7ff;overflow-wrap:anywhere">{html.escape(str(result["recovered_message"]))}</div></div>',
                unsafe_allow_html=True,
            )
            if result["success"]:
                st.caption("Original and recovered message match.")
            else:
                st.caption("A mismatch is an expected possible outcome at low SNR.")

        first = result["blocks"][0]
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.stem(np.arange(N), first["received"], basefmt=" ", linefmt="#67e8f9", markerfmt="o")
        ax.axhline(0, color="#a78bfa", linestyle="--", linewidth=1)
        ax.set_xlabel("BPSK symbol index")
        ax.set_ylabel("Received amplitude")
        ax.set_title("Received channel samples · first Polar block")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Monte-Carlo performance lab
# -----------------------------------------------------------------------------
with performance_tab:
    section_head("04", "Performance lab", "Run a Monte-Carlo SNR sweep and measure the reliability of the Polar-coded link.")
    st.markdown(
        """<div class="muted-box" style="margin-bottom:1rem">
        <div class="mini-label">EXPERIMENT DESIGN</div>
        <div style="color:#dceaff;margin-top:.45rem">For each Eb/N0 point from 0 to 15 dB, random information blocks are encoded, modulated, passed through AWGN, and decoded with SC. BER and FER are accumulated over all frames.</div>
        </div>""",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        p1, p2 = st.columns([1, 1])
        with p1:
            frames = st.slider("Frames per SNR point", min_value=200, max_value=5000, value=1200, step=200)
        with p2:
            perf_seed = st.number_input("Simulation seed", min_value=0, max_value=999999, value=2026, step=1, key="perf_seed")
        st.caption(f"{frames:,} frames × {K} information bits = {frames*K:,} information bits tested at each SNR.")
        run_perf = st.button("⏵  Run 0–15 dB experiment", type="primary", use_container_width=True)

    if run_perf:
        with st.spinner("Running simulation across 16 SNR points…"):
            st.session_state["performance_result"] = simulate_ber(
                range(16), frames_per_snr=int(frames), seed=int(perf_seed)
            )

    perf = st.session_state.get("performance_result")
    if perf is not None:
        st.markdown(" ")
        section_head("05", "Measured performance", "The curves show the measured error rates; zero-count points use the one-error measurement floor for logarithmic plotting.")
        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            metric_tile("SNR POINTS", "16", "0 dB through 15 dB")
        with pm2:
            metric_tile("FRAMES / POINT", f"{perf['frames_per_snr']:,}", "Monte-Carlo trials")
        with pm3:
            metric_tile("BITS / POINT", f"{perf['total_bits_per_snr']:,}", "Information bits")

        ber_floor = 1.0 / perf["total_bits_per_snr"]
        fer_floor = 1.0 / perf["frames_per_snr"]
        ber_plot = np.where(perf["ber"] == 0, ber_floor, perf["ber"])
        fer_plot = np.where(perf["fer"] == 0, fer_floor, perf["fer"])

        plot1, plot2 = st.columns(2, gap="large")
        with plot1:
            fig1, ax1 = plt.subplots(figsize=(7.2, 4.5))
            ax1.semilogy(perf["snr_db"], ber_plot, marker="o", markersize=5, linewidth=2.2, color="#67e8f9", label="Polar code + SC")
            ax1.semilogy(perf["snr_db"], perf["uncoded_ber"], linestyle="--", linewidth=1.7, color="#a78bfa", label="Uncoded BPSK theory")
            ax1.set_xlabel("Eb/N0 (dB)")
            ax1.set_ylabel("Bit Error Rate")
            ax1.set_title("BER vs Eb/N0")
            ax1.legend(facecolor="#14213a", edgecolor="#2a3a57", labelcolor="#dce8fa", fontsize=8)
            style_axis(ax1)
            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
        with plot2:
            fig2, ax2 = plt.subplots(figsize=(7.2, 4.5))
            ax2.semilogy(perf["snr_db"], fer_plot, marker="s", markersize=5, linewidth=2.2, color="#6ee7b7", label="Polar code + SC")
            ax2.set_xlabel("Eb/N0 (dB)")
            ax2.set_ylabel("Frame Error Rate")
            ax2.set_title("FER vs Eb/N0")
            ax2.legend(facecolor="#14213a", edgecolor="#2a3a57", labelcolor="#dce8fa", fontsize=8)
            style_axis(ax2)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        perf_rows = []
        for i, snr in enumerate(perf["snr_db"]):
            perf_rows.append({
                "Eb/N0 (dB)": int(snr),
                "Bit errors": int(perf["bit_errors"][i]),
                "Frame errors": int(perf["frame_errors"][i]),
                "BER": f"{perf['ber'][i]:.7g}",
                "FER": f"{perf['fer'][i]:.7g}",
            })
        with st.expander("View complete numerical results", expanded=True):
            st.dataframe(perf_rows, use_container_width=True, hide_index=True)
        if np.any(perf["bit_errors"] == 0):
            st.info("Zero observed errors means no errors occurred in the simulated frames; it does not prove the true BER/FER is zero. The graph places zero-count points at the one-error measurement floor.")


# -----------------------------------------------------------------------------
# Theory and model
# -----------------------------------------------------------------------------
with theory_tab:
    section_head("06", "Theory & system model", "The mathematical model behind the transmitter, AWGN channel and successive-cancellation receiver.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### 1 · Polar transform")
            st.latex(r"F = \begin{bmatrix}1&0\\1&1\end{bmatrix}")
            st.latex(r"G_N = F^{\otimes m}, \qquad N=2^m")
            st.latex(r"x = uG_N \pmod 2")
            st.caption("The input vector is transformed over GF(2), where addition is XOR.")
        with st.container(border=True):
            st.markdown("### 2 · BPSK modulation")
            st.latex(r"0 \mapsto +1, \qquad 1 \mapsto -1")
            st.caption("Each coded bit is mapped to one real-valued channel symbol.")
        with st.container(border=True):
            st.markdown("### 3 · AWGN channel")
            st.latex(r"y=s+n, \qquad n\sim\mathcal{N}(0,\sigma^2)")
            st.latex(r"\sigma^2=\frac{1}{2R\cdot10^{(E_b/N_0)_{\mathrm{dB}}/10}}")
            st.latex(r"R=\frac{K}{N}")
            st.caption("Noise variance decreases as Eb/N0 increases. SNR is specified per information bit.")
    with right:
        with st.container(border=True):
            st.markdown("### 4 · Channel log-likelihood ratio")
            st.latex(r"L(y)=\log\frac{p(y\mid x=0)}{p(y\mid x=1)}=\frac{2y}{\sigma^2}")
            st.caption("With this BPSK mapping, positive LLR values favour bit 0.")
        with st.container(border=True):
            st.markdown("### 5 · Code construction")
            st.latex(r"N=8,\qquad K=4,\qquad R=\frac{K}{N}=\frac{1}{2}")
            st.write(f"**Information positions:** `{INFO_POS.tolist()}`")
            st.write(f"**Frozen positions:** `{FROZEN_POS.tolist()}`")
            st.warning("This is a small educational Polar code construction, not a 5G NR-compliant construction.")
        with st.container(border=True):
            st.markdown("### 6 · Successive Cancellation decoding")
            st.write("The recursive decoder applies an f update to decode the first half, followed by a g update that uses previously decoded partial sums for the second half. Frozen-bit decisions are forced to zero.")
            st.latex(r"f(a,b)=2\tanh^{-1}\!\left(\tanh\frac{a}{2}\tanh\frac{b}{2}\right)")
            st.latex(r"g(a,b,\hat{u})=b+(1-2\hat{u})a")
    with st.container(border=True):
        st.markdown("### 7 · Text framing and message reconstruction")
        st.write("The UTF-8 payload is preceded by a 16-bit payload-length header. The framed stream is divided into 4-bit information blocks; each block is encoded into an 8-bit Polar codeword. At the receiver, decoded blocks are concatenated, the header is read, padding is removed, and the UTF-8 byte sequence is reconstructed.")


st.markdown(
    """<div class="footer">POLAR CODE · EDUCATIONAL SIMULATION · N=8 / K=4 · BPSK OVER AWGN <br>
            Made by KSHITIZ SINHA </div>""",
    unsafe_allow_html=True,
)