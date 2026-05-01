---
title: Music Source Unmixing
emoji: 💻
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: Simple music source unmixing
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Music Source Unmixing

This repository contains an interactive music source unmixing mini app.

The app separates a music mixture into four stems:

- vocals
- drums
- bass
- other

It is designed as a compact educational and portfolio demo for audio source separation, stem analysis, and remixing.

A Streamlit deployment is available here:

https://music-source-unmixing.streamlit.app/

## Main features

- Upload an audio file or use the default demo audio.
- Separate a music mixture into four standard stems:
  - vocals,
  - drums,
  - bass,
  - other.
- Choose between two Demucs models:
  - `htdemucs_ft`,
  - `mdx_extra`.
- Select the execution device:
  - `auto`,
  - `cpu`.
- Listen to each separated stem directly in the interface.
- Download the separated stems as a ZIP archive.
- Display a 2 by 4 analysis figure:
  - row 1: spectrograms of the four stems,
  - row 2: magnitude spectra of the four stems.
- Modify each stem gain in dB.
- Build and listen to a custom remix.
- Read the English and French documentation tabs.

## Method overview

The app relies on Demucs for deep-learning-based music source separation.

Given an input mixture, the selected model estimates four audio sources corresponding to vocals, drums, bass, and other instruments. The separated signals are then saved as WAV files and made available through audio players and a downloadable ZIP archive.

The app also computes two complementary visual analyses for each stem.

The spectrogram gives a time-frequency representation of the signal, showing how the frequency content evolves over time.

The magnitude spectrum gives a global frequency-domain view of each separated source.

## Available models

### `htdemucs_ft`

`htdemucs_ft` is a fine-tuned Hybrid Transformer Demucs model.

It is a strong default choice when quality is the priority.

### `mdx_extra`

`mdx_extra` is an MDX-based model available through Demucs.

It provides an alternative separation behavior and can be useful for comparison.

## Remix controls

After separation, each stem can be amplified or attenuated using a gain slider expressed in dB.

The available range is:

```text
-60 dB to +12 dB
```

This makes it possible to suppress a stem almost completely, reduce its presence, or slightly boost it before building a remix.

The remix is normalised when necessary to avoid clipping.

## Repository structure

```text
.
├── app.py                 # Gradio / Hugging Face Space entry point
├── app_sl.py              # Streamlit version of the app
├── documentation_en.md    # English documentation
├── documentation_fr.md    # French documentation
├── requirements.txt       # Python dependencies
├── LICENSE.txt            # License file
└── README.md              # Repository and Hugging Face Space description
```

## Installation

Clone the repository:

```bash
git clone https://github.com/trungtin-dinh/music_source_unmixing.git
cd music_source_unmixing
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The app uses Demucs, PyTorch, torchaudio, Plotly, SoundFile, NumPy, Gradio, and Streamlit.

## Run the Gradio app

```bash
python app.py
```

The local interface will usually be available at:

```text
http://127.0.0.1:7860
```

## Run the Streamlit app

```bash
streamlit run app_sl.py
```

The local interface will usually be available at:

```text
http://localhost:8501
```

## Hugging Face Space notes

The YAML block at the top of this README is used by Hugging Face Spaces.

The current metadata launches the Gradio version:

```yaml
sdk: gradio
app_file: app.py
```

If you want Hugging Face to launch the Streamlit version instead, update the metadata to:

```yaml
sdk: streamlit
app_file: app_sl.py
```

In that case, make sure `streamlit` is included in `requirements.txt`.

## Documentation

The repository includes two Markdown documentation files:

- `documentation_en.md` for the English documentation.
- `documentation_fr.md` for the French documentation.

These files explain the music source separation problem, classical approaches such as ICA and NMF, time-frequency analysis with the STFT, the magnitude spectrum, deep-learning-based source separation, Demucs, stem remixing, and practical limitations.

## Notes on performance

Music source separation can be computationally expensive.

On CPU, separation may take time depending on the length of the input audio and the selected model.

For online deployment, short audio examples are recommended to keep execution time reasonable.

## License

This project is released under the MIT License.

## Author

Developed by Trung-Tin Dinh as part of a portfolio of interactive signal, audio, image, and computer vision mini apps.
