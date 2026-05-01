import shutil
import subprocess
import sys
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import torch
from plotly.subplots import make_subplots


LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

with open("documentation_fr.md", "r", encoding="utf-8") as f:
    DOCUMENTATION_fr = f.read()

with open("documentation_en.md", "r", encoding="utf-8") as f:
    DOCUMENTATION_en = f.read()


def split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections = {}
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

    return sections


DOC_FR_SECTIONS = split_markdown_by_h2(DOCUMENTATION_fr)
DOC_EN_SECTIONS = split_markdown_by_h2(DOCUMENTATION_en)

DOC_FR_TITLES = list(DOC_FR_SECTIONS.keys())
DOC_EN_TITLES = list(DOC_EN_SECTIONS.keys())


def load_doc_fr_section(title: str) -> str:
    return DOC_FR_SECTIONS[title]


def load_doc_en_section(title: str) -> str:
    return DOC_EN_SECTIONS[title]


DEFAULT_AUDIO_URL = "https://raw.githubusercontent.com/pdx-cs-sound/wavs/main/collectathon.wav"

MODEL_CHOICES = [
    "htdemucs_ft",
    "mdx_extra",
]

STEM_NAMES = ["vocals", "drums", "bass", "other"]

MIN_GAIN_DB = -60.0
MAX_GAIN_DB = 12.0
GAIN_STEP_DB = 0.5


def resolve_device(device_choice: str) -> str:
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_choice


def db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (gain_db / 20.0))


def prepare_input_audio(audio_source: str, work_dir: Path) -> Path:
    parsed = urlparse(audio_source)
    is_remote = parsed.scheme in {"http", "https"}

    if is_remote:
        suffix = Path(parsed.path).suffix or ".wav"
        local_path = work_dir / f"input_audio{suffix}"
        urlretrieve(audio_source, local_path)
        return local_path

    source_path = Path(audio_source)
    suffix = source_path.suffix or ".wav"
    local_path = work_dir / f"input_audio{suffix}"
    shutil.copy2(source_path, local_path)
    return local_path


def find_track_output_dir(work_dir: Path, model_name: str, track_stem: str) -> Path:
    expected = work_dir / "separated" / model_name / track_stem
    if expected.exists():
        return expected

    model_dir = work_dir / "separated" / model_name
    if not model_dir.exists():
        raise FileNotFoundError("Demucs output folder was not created.")

    subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]

    raise FileNotFoundError("Could not locate the separated stems folder.")


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
        "Vocals – spectrogram",
        "Drums – spectrogram",
        "Bass – spectrogram",
        "Other – spectrogram",
        "Vocals – magnitude spectrum",
        "Drums – magnitude spectrum",
        "Bass – magnitude spectrum",
        "Other – magnitude spectrum",
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


def separate_audio(
    audio_path: str | None,
    model_name: str,
    device_choice: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, str, dict[str, Any], go.Figure]:
    if not audio_path:
        raise gr.Error("Please upload an audio file.")

    progress(0.05, desc="Preparing audio")

    work_dir = Path(tempfile.mkdtemp(prefix="demucs_space_"))
    input_copy = prepare_input_audio(audio_path, work_dir)

    device = resolve_device(device_choice)

    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model_name,
        "-d",
        device,
        str(input_copy),
    ]

    progress(0.15, desc="Running separation")
    completed = subprocess.run(
        command,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "Unknown Demucs error").strip()
        raise gr.Error(details)

    progress(0.85, desc="Collecting stems")

    track_dir = find_track_output_dir(work_dir, model_name, input_copy.stem)
    stem_paths = {name: track_dir / f"{name}.wav" for name in STEM_NAMES}

    missing = [name for name, path in stem_paths.items() if not path.exists()]
    if missing:
        raise gr.Error(f"Missing output stems: {', '.join(missing)}")

    zip_path = create_zip_file(list(stem_paths.values()), work_dir / "separated_stems.zip")

    state = {
        "work_dir": str(work_dir),
        "source": str(input_copy),
        "stems": {name: str(path) for name, path in stem_paths.items()},
    }

    figure = build_analysis_figure(
        {
            "vocals": str(stem_paths["vocals"]),
            "drums": str(stem_paths["drums"]),
            "bass": str(stem_paths["bass"]),
            "other": str(stem_paths["other"]),
        }
    )

    progress(1.0, desc="Finished")

    return (
        str(stem_paths["vocals"]),
        str(stem_paths["drums"]),
        str(stem_paths["bass"]),
        str(stem_paths["other"]),
        str(zip_path),
        state,
        figure,
    )


def remix_stems(
    state: dict[str, Any] | None,
    vocals_gain_db: float,
    drums_gain_db: float,
    bass_gain_db: float,
    other_gain_db: float,
) -> str:
    if not state or "stems" not in state:
        raise gr.Error("Please separate an audio file first.")

    gains = {
        "vocals": db_to_linear(vocals_gain_db),
        "drums": db_to_linear(drums_gain_db),
        "bass": db_to_linear(bass_gain_db),
        "other": db_to_linear(other_gain_db),
    }

    stems = state["stems"]
    mixed_signal = None
    sample_rate = None

    for stem_name in STEM_NAMES:
        audio, sr = sf.read(stems[stem_name], always_2d=True, dtype="float32")

        if sample_rate is None:
            sample_rate = sr
            mixed_signal = np.zeros_like(audio, dtype=np.float32)
        elif sr != sample_rate:
            raise gr.Error("Sample-rate mismatch between stems.")

        mixed_signal += gains[stem_name] * audio

    peak = float(np.max(np.abs(mixed_signal)))
    if peak > 1.0:
        mixed_signal /= peak

    remix_path = Path(state["work_dir"]) / "remix.wav"
    sf.write(remix_path, mixed_signal, sample_rate)
    return str(remix_path)


with gr.Blocks(title="Music Source Separation") as demo:
    state = gr.State()

    with gr.Tab("App"):
        with gr.Row():
            audio_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                value=DEFAULT_AUDIO_URL,
                label="Input audio",
                scale=4,
            )
            model_name = gr.Dropdown(
                choices=MODEL_CHOICES,
                value="htdemucs_ft",
                label="Model",
                scale=1,
            )
            device_choice = gr.Dropdown(
                choices=["auto", "cpu"],
                value="auto",
                label="Device",
                scale=1,
            )
            run_button = gr.Button("Separate", variant="primary", scale=1)

        with gr.Row():
            vocals_audio = gr.Audio(label="Vocals")
            drums_audio = gr.Audio(label="Drums")
            bass_audio = gr.Audio(label="Bass")
            other_audio = gr.Audio(label="Other")

        stems_zip = gr.File(label="Download stems (.zip)")

        analysis_plot = gr.Plot(label="Stem Analysis")

        with gr.Row():
            vocals_gain = gr.Slider(
                minimum=MIN_GAIN_DB,
                maximum=MAX_GAIN_DB,
                value=0.0,
                step=GAIN_STEP_DB,
                label="Vocals gain (dB)",
            )
            drums_gain = gr.Slider(
                minimum=MIN_GAIN_DB,
                maximum=MAX_GAIN_DB,
                value=0.0,
                step=GAIN_STEP_DB,
                label="Drums gain (dB)",
            )
            bass_gain = gr.Slider(
                minimum=MIN_GAIN_DB,
                maximum=MAX_GAIN_DB,
                value=0.0,
                step=GAIN_STEP_DB,
                label="Bass gain (dB)",
            )
            other_gain = gr.Slider(
                minimum=MIN_GAIN_DB,
                maximum=MAX_GAIN_DB,
                value=0.0,
                step=GAIN_STEP_DB,
                label="Other gain (dB)",
            )

        with gr.Row():
            remix_button = gr.Button("Build remix", variant="primary", scale=1)
            remix_audio = gr.Audio(label="Remix", scale=5)

        run_button.click(
            fn=separate_audio,
            inputs=[audio_input, model_name, device_choice],
            outputs=[
                vocals_audio,
                drums_audio,
                bass_audio,
                other_audio,
                stems_zip,
                state,
                analysis_plot,
            ],
        )

        remix_button.click(
            fn=remix_stems,
            inputs=[state, vocals_gain, drums_gain, bass_gain, other_gain],
            outputs=remix_audio,
        )

    with gr.Tab("Documentation FR"):
        with gr.Row():
            with gr.Column(scale=1):
                doc_fr_buttons = []
                for title in DOC_FR_TITLES:
                    btn = gr.Button(title)
                    doc_fr_buttons.append((btn, title))

            with gr.Column(scale=2):
                doc_fr_view = gr.Markdown(
                    value=load_doc_fr_section(DOC_FR_TITLES[0]),
                    latex_delimiters=LATEX_DELIMITERS,
                )

        for btn, title in doc_fr_buttons:
            btn.click(
                lambda t=title: load_doc_fr_section(t),
                inputs=None,
                outputs=doc_fr_view,
            )

    with gr.Tab("Documentation EN"):
        with gr.Row():
            with gr.Column(scale=1):
                doc_en_buttons = []
                for title in DOC_EN_TITLES:
                    btn = gr.Button(title)
                    doc_en_buttons.append((btn, title))

            with gr.Column(scale=2):
                doc_en_view = gr.Markdown(
                    value=load_doc_en_section(DOC_EN_TITLES[0]),
                    latex_delimiters=LATEX_DELIMITERS,
                )

        for btn, title in doc_en_buttons:
            btn.click(
                lambda t=title: load_doc_en_section(t),
                inputs=None,
                outputs=doc_en_view,
            )

demo.queue()
demo.launch()
