import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
import torch
from plotly.subplots import make_subplots


DEFAULT_AUDIO_URL = "https://raw.githubusercontent.com/pdx-cs-sound/wavs/main/collectathon.wav"

MODEL_CHOICES = [
    "htdemucs_ft",
    "mdx_extra",
]

STEM_NAMES = ["vocals", "drums", "bass", "other"]

MIN_GAIN_DB = -60.0
MAX_GAIN_DB = 12.0
GAIN_STEP_DB = 0.5

DOC_FILE_FR = "documentation_fr.md"
DOC_FILE_EN = "documentation_en.md"


st.set_page_config(
    page_title="Music Source Separation",
    layout="wide",
)


st.markdown(
    """
    <style>
    /* Push the whole interface below Streamlit's fixed top bar. */
    .block-container {
        padding-top: 4.8rem;
        padding-bottom: 2rem;
    }

    /* Keep buttons as rectangular Gradio-like boxes. */
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        min-height: 2.65rem;
        border-radius: 8px;
        white-space: normal;
    }

    /* Make selected documentation buttons look active, like Gradio's selected box. */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        font-weight: 700;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.24);
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }

    div[data-testid="stFileUploader"] section {
        min-height: 5.6rem;
    }

    div[data-testid="stAudio"] {
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_markdown_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return (
            f"## Documentation file not found\n\n"
            f"The file `{path}` was not found next to `app_sl.py`. "
            f"Place `{DOC_FILE_FR}` and `{DOC_FILE_EN}` in the same folder as the app."
        )
    return file_path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    parts = re.split(r"(?m)^##\s+", markdown_text.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        title = lines[0].strip()

        if title.lower() in {"table des matières", "table of contents"}:
            continue

        sections[title] = "## " + part

    if not sections:
        sections["Documentation"] = markdown_text

    return sections


def resolve_device(device_choice: str) -> str:
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_choice


def db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (gain_db / 20.0))


def save_uploaded_audio(uploaded_file: Any, work_dir: Path) -> Path:
    original_name = Path(uploaded_file.name).name
    suffix = Path(original_name).suffix or ".wav"
    local_path = work_dir / f"input_audio{suffix}"

    with local_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    return local_path


def download_audio_from_url(audio_url: str, work_dir: Path) -> Path:
    parsed = urlparse(audio_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("The default audio URL is not a valid HTTP/HTTPS URL.")

    suffix = Path(parsed.path).suffix or ".wav"
    local_path = work_dir / f"input_audio{suffix}"
    urlretrieve(audio_url, local_path)
    return local_path


def prepare_input_audio(uploaded_file: Any | None, work_dir: Path) -> Path:
    if uploaded_file is not None:
        return save_uploaded_audio(uploaded_file, work_dir)
    return download_audio_from_url(DEFAULT_AUDIO_URL, work_dir)


def create_zip_file(files: list[Path], zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            if file_path.exists():
                zf.write(file_path, arcname=file_path.name)
    return zip_path


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    signal, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    signal = signal.mean(axis=1)
    return signal.astype(np.float32), sample_rate


def downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = int(np.ceil(len(x) / max_points))
    return x[::step], y[::step]


def compute_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(signal) < n_fft:
        pad_width = n_fft - len(signal)
        signal = np.pad(signal, (0, pad_width), mode="constant")

    waveform = torch.from_numpy(signal)
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = torch.abs(stft).numpy()
    magnitude = np.log1p(magnitude)

    freqs = np.linspace(0.0, sample_rate / 2.0, magnitude.shape[0], dtype=np.float32)
    times = np.arange(magnitude.shape[1], dtype=np.float32) * hop_length / float(sample_rate)
    return times, freqs, magnitude


def compute_frequency_spectrum(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    if len(signal) == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    magnitude = np.abs(np.fft.rfft(signal)) / max(1, len(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)

    freqs, magnitude = downsample_curve(
        freqs.astype(np.float32),
        magnitude.astype(np.float32),
        max_points=5000,
    )
    return freqs, magnitude


def build_analysis_figure(audio_paths: dict[str, str]) -> go.Figure:
    subplot_titles = [
        "Vocals - spectrogram",
        "Drums - spectrogram",
        "Bass - spectrogram",
        "Other - spectrogram",
        "Vocals - magnitude spectrum",
        "Drums - magnitude spectrum",
        "Bass - magnitude spectrum",
        "Other - magnitude spectrum",
    ]

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.14,
    )

    for col_index, stem_name in enumerate(STEM_NAMES, start=1):
        signal, sample_rate = load_audio_mono(audio_paths[stem_name])

        spec_times, spec_freqs, spec_mag = compute_spectrogram(signal, sample_rate)
        fig.add_trace(
            go.Heatmap(
                x=spec_times,
                y=spec_freqs,
                z=spec_mag,
                showscale=False,
                hovertemplate="Time: %{x:.2f} s<br>Freq: %{y:.0f} Hz<br>Value: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=col_index,
        )

        spec_freqs_curve, spec_mag_curve = compute_frequency_spectrum(signal, sample_rate)
        fig.add_trace(
            go.Scattergl(
                x=spec_freqs_curve,
                y=spec_mag_curve,
                mode="lines",
                showlegend=False,
                hovertemplate="Freq: %{x:.0f} Hz<br>Magnitude: %{y:.6f}<extra></extra>",
            ),
            row=2,
            col=col_index,
        )

        fig.update_xaxes(title_text="Time (s)", row=1, col=col_index)
        fig.update_yaxes(title_text="Freq (Hz)", row=1, col=col_index)
        fig.update_xaxes(title_text="Freq (Hz)", row=2, col=col_index)
        fig.update_yaxes(title_text="Magnitude", row=2, col=col_index)

    fig.update_layout(height=760, margin=dict(l=40, r=20, t=60, b=40))
    return fig


@st.cache_resource(show_spinner=False)
def load_demucs_model(model_name: str, device: str) -> Any:
    try:
        from demucs.pretrained import get_model
    except Exception as exc:
        raise RuntimeError(
            "Demucs is not installed or cannot be imported. Install it with `pip install demucs`."
        ) from exc

    model = get_model(name=model_name)
    model.to(device)
    model.eval()
    return model


def save_stem_with_soundfile(stem_tensor: torch.Tensor, output_path: Path, sample_rate: int) -> None:
    audio = stem_tensor.detach().cpu().float().numpy()

    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2:
        audio = audio.T
    else:
        raise ValueError("Unexpected stem tensor shape returned by Demucs.")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    sf.write(output_path, audio, sample_rate)


def run_demucs_in_process(
    input_path: Path,
    model_name: str,
    device: str,
    output_dir: Path,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Path]:
    try:
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
    except Exception as exc:
        raise RuntimeError(
            "Demucs is not installed or cannot be imported. Install it with `pip install demucs`."
        ) from exc

    def update_progress(value: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    device_obj = torch.device(device)

    update_progress(15, "Loading Demucs model")
    model = load_demucs_model(model_name, device)
    model.to(device_obj)
    model.eval()

    update_progress(25, "Reading input audio")
    wav = AudioFile(str(input_path)).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    wav = wav.to(device_obj)

    ref = wav.mean(0)
    mean = ref.mean()
    std = ref.std()
    if float(std.detach().cpu()) < 1e-8:
        std = torch.tensor(1.0, device=device_obj, dtype=wav.dtype)

    wav = (wav - mean) / std

    update_progress(35, "Running Demucs separation")
    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            device=device_obj,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]

    sources = sources * std + mean
    source_names = list(model.sources)

    missing_sources = [stem for stem in STEM_NAMES if stem not in source_names]
    if missing_sources:
        raise RuntimeError(f"The selected Demucs model did not return these stems: {', '.join(missing_sources)}")

    update_progress(80, "Saving stems")
    track_dir = output_dir / "separated" / model_name / input_path.stem
    track_dir.mkdir(parents=True, exist_ok=True)

    stem_paths: dict[str, Path] = {}
    for stem_name in STEM_NAMES:
        source_index = source_names.index(stem_name)
        output_path = track_dir / f"{stem_name}.wav"
        save_stem_with_soundfile(sources[source_index], output_path, int(model.samplerate))
        stem_paths[stem_name] = output_path

    return stem_paths


def separate_audio(
    uploaded_file: Any | None,
    model_name: str,
    device_choice: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[dict[str, Any], go.Figure]:
    work_dir = Path(tempfile.mkdtemp(prefix="demucs_streamlit_"))
    input_copy = prepare_input_audio(uploaded_file, work_dir)
    device = resolve_device(device_choice)

    stem_paths = run_demucs_in_process(
        input_path=input_copy,
        model_name=model_name,
        device=device,
        output_dir=work_dir,
        progress_callback=progress_callback,
    )

    missing = [name for name, path in stem_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing output stems: {', '.join(missing)}")

    zip_path = create_zip_file(list(stem_paths.values()), work_dir / "separated_stems.zip")

    state = {
        "work_dir": str(work_dir),
        "source": str(input_copy),
        "zip_path": str(zip_path),
        "stems": {name: str(path) for name, path in stem_paths.items()},
    }

    figure = build_analysis_figure(state["stems"])
    return state, figure


def remix_stems(
    state: dict[str, Any] | None,
    vocals_gain_db: float,
    drums_gain_db: float,
    bass_gain_db: float,
    other_gain_db: float,
) -> str:
    if not state or "stems" not in state:
        raise ValueError("Please separate an audio file first.")

    gains = {
        "vocals": db_to_linear(vocals_gain_db),
        "drums": db_to_linear(drums_gain_db),
        "bass": db_to_linear(bass_gain_db),
        "other": db_to_linear(other_gain_db),
    }

    stems = state["stems"]
    mixed_signal: np.ndarray | None = None
    sample_rate: int | None = None

    for stem_name in STEM_NAMES:
        audio, sr = sf.read(stems[stem_name], always_2d=True, dtype="float32")

        if sample_rate is None:
            sample_rate = sr
            mixed_signal = np.zeros_like(audio, dtype=np.float32)
        elif sr != sample_rate:
            raise ValueError("Sample-rate mismatch between stems.")

        if mixed_signal is None:
            raise ValueError("Could not initialise the remix buffer.")

        mixed_signal += gains[stem_name] * audio

    if mixed_signal is None or sample_rate is None:
        raise ValueError("No stems were available for remixing.")

    peak = float(np.max(np.abs(mixed_signal)))
    if peak > 1.0:
        mixed_signal /= peak

    remix_path = Path(state["work_dir"]) / "remix.wav"
    sf.write(remix_path, mixed_signal, sample_rate)
    return str(remix_path)


def render_audio_player(label: str, path: str | None) -> None:
    st.caption(label)
    if path and Path(path).exists():
        st.audio(Path(path).read_bytes(), format="audio/wav")
    else:
        st.info("No audio yet.")


def render_download_button(label: str, path: str | None, file_name: str, mime: str) -> None:
    if path and Path(path).exists():
        st.download_button(
            label=label,
            data=Path(path).read_bytes(),
            file_name=file_name,
            mime=mime,
            use_container_width=True,
        )
    else:
        st.download_button(
            label=label,
            data=b"",
            file_name=file_name,
            mime=mime,
            disabled=True,
            use_container_width=True,
        )


def ensure_session_state() -> None:
    st.session_state.setdefault("separation_state", None)
    st.session_state.setdefault("analysis_figure", None)
    st.session_state.setdefault("remix_path", None)


def render_input_audio_preview(uploaded_file: Any | None) -> None:
    if uploaded_file is None:
        st.caption("Default audio")
        st.audio(DEFAULT_AUDIO_URL)
    else:
        st.caption("Uploaded audio")
        st.audio(uploaded_file.getvalue(), format=uploaded_file.type or "audio/wav")


def render_app_tab() -> None:
    ensure_session_state()

    input_col, model_col, device_col, button_col = st.columns([4, 1.25, 1.0, 1.0])

    with input_col:
        uploaded_file = st.file_uploader(
            "Input audio",
            type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
        )
        render_input_audio_preview(uploaded_file)

    with model_col:
        model_name = st.selectbox("Model", MODEL_CHOICES, index=0)

    with device_col:
        device_choice = st.selectbox("Device", ["auto", "cpu"], index=0)

    with button_col:
        st.write("")
        st.write("")
        separate_clicked = st.button("Separate", type="primary", use_container_width=True)

    if separate_clicked:
        progress_bar = st.progress(0, text="Preparing audio")

        def update_progress(value: int, message: str) -> None:
            progress_bar.progress(value, text=message)

        try:
            update_progress(5, "Preparing audio")
            state, figure = separate_audio(
                uploaded_file=uploaded_file,
                model_name=model_name,
                device_choice=device_choice,
                progress_callback=update_progress,
            )
            update_progress(90, "Building stem analysis")
            st.session_state["separation_state"] = state
            st.session_state["analysis_figure"] = figure
            st.session_state["remix_path"] = None
            update_progress(100, "Finished")
            st.success("Separation finished.")
        except Exception as exc:
            st.session_state["separation_state"] = None
            st.session_state["analysis_figure"] = None
            st.session_state["remix_path"] = None
            progress_bar.empty()
            st.error(str(exc))

    state = st.session_state.get("separation_state")
    stems = state.get("stems", {}) if state else {}

    audio_cols = st.columns(4)
    for col, stem_name, label in zip(audio_cols, STEM_NAMES, ["Vocals", "Drums", "Bass", "Other"]):
        with col:
            render_audio_player(label, stems.get(stem_name))

    render_download_button(
        label="Download stems (.zip)",
        path=state.get("zip_path") if state else None,
        file_name="separated_stems.zip",
        mime="application/zip",
    )

    if st.session_state.get("analysis_figure") is not None:
        st.plotly_chart(st.session_state["analysis_figure"], use_container_width=True)
    else:
        st.info("Run the separation to display the stem spectrograms and magnitude spectra.")

    gain_cols = st.columns(4)
    with gain_cols[0]:
        vocals_gain = st.slider("Vocals gain (dB)", MIN_GAIN_DB, MAX_GAIN_DB, 0.0, GAIN_STEP_DB)
    with gain_cols[1]:
        drums_gain = st.slider("Drums gain (dB)", MIN_GAIN_DB, MAX_GAIN_DB, 0.0, GAIN_STEP_DB)
    with gain_cols[2]:
        bass_gain = st.slider("Bass gain (dB)", MIN_GAIN_DB, MAX_GAIN_DB, 0.0, GAIN_STEP_DB)
    with gain_cols[3]:
        other_gain = st.slider("Other gain (dB)", MIN_GAIN_DB, MAX_GAIN_DB, 0.0, GAIN_STEP_DB)

    remix_button_col, remix_audio_col = st.columns([1, 5])
    with remix_button_col:
        st.write("")
        build_remix_clicked = st.button(
            "Build remix",
            type="primary",
            disabled=state is None,
            use_container_width=True,
        )

    if build_remix_clicked:
        try:
            remix_path = remix_stems(
                state,
                vocals_gain,
                drums_gain,
                bass_gain,
                other_gain,
            )
            st.session_state["remix_path"] = remix_path
            st.success("Remix built.")
        except Exception as exc:
            st.session_state["remix_path"] = None
            st.error(str(exc))

    with remix_audio_col:
        render_audio_player("Remix", st.session_state.get("remix_path"))


def select_doc_section(state_key: str, title: str) -> None:
    st.session_state[state_key] = title


def render_documentation_tab(markdown_text: str, state_key: str) -> None:
    sections = split_markdown_by_h2(markdown_text)
    titles = list(sections.keys())

    if not titles:
        st.warning("No documentation section was found.")
        return

    if state_key not in st.session_state or st.session_state[state_key] not in sections:
        st.session_state[state_key] = titles[0]

    nav_col, doc_col = st.columns([1, 2])

    with nav_col:
        for title in titles:
            is_selected = st.session_state[state_key] == title
            button_type = "primary" if is_selected else "secondary"
            st.button(
                title,
                key=f"{state_key}_{title}",
                type=button_type,
                use_container_width=True,
                on_click=select_doc_section,
                args=(state_key, title),
            )

    selected_title = st.session_state[state_key]
    with doc_col:
        st.markdown(sections[selected_title])


def main() -> None:
    tab_app, tab_doc_fr, tab_doc_en = st.tabs(["App", "Documentation FR", "Documentation EN"])

    with tab_app:
        render_app_tab()

    with tab_doc_fr:
        render_documentation_tab(read_markdown_file(DOC_FILE_FR), "selected_doc_fr")

    with tab_doc_en:
        render_documentation_tab(read_markdown_file(DOC_FILE_EN), "selected_doc_en")


if __name__ == "__main__":
    main()
