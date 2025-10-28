from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import wave

import genie_tts as genie

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REVIEW_GEN_AUDIO = False  # Play back generated lines after synthesis

PROJECT_ROOT = Path(__file__).resolve().parent
SOVITS_ROOT = PROJECT_ROOT / "SoVITSModel"  # Each character has a subfolder here
GENIE_ROOT = PROJECT_ROOT / "GenieModel"

CSV_PATH = PROJECT_ROOT / "Dialog" / "dialog.csv"  # CSV format: Chara, Dialog
OUTPUT_ROOT = PROJECT_ROOT / "output"
SCRIPT_NAME = CSV_PATH.stem

REQUIRED_ONNX_FILES: tuple[str, ...] = (
    "t2s_encoder_fp32.onnx",
    "t2s_first_stage_decoder_fp32.onnx",
    "t2s_stage_decoder_fp32.onnx",
    "vits_fp32.onnx",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CharacterAssets:
    model_dir: Path
    reference_audio: Path
    reference_text: str
    ckpt_path: Optional[Path] = None
    pth_path: Optional[Path] = None
    prebuilt: bool = False


@dataclass
class CharacterSession:
    display_name: str
    key: str
    model_dir: Path
    assets: CharacterAssets
    line_counter: int = 0


def find_character_dir(root: Path, character_name: str) -> Optional[Path]:
    """Find a case-insensitive character directory within the given root."""
    direct_path = root / character_name
    if direct_path.is_dir():
        return direct_path

    target_lower = character_name.lower()
    if root.is_dir():
        for candidate in root.iterdir():
            if candidate.is_dir() and candidate.name.lower() == target_lower:
                return candidate
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    """Configure module-level logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def ensure_directories(paths: Iterable[Path]) -> None:
    """Ensure all provided directories exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def has_onnx_bundle(model_dir: Path) -> bool:
    """Check whether all required ONNX files exist inside model_dir."""
    return all((model_dir / filename).is_file() for filename in REQUIRED_ONNX_FILES)


def ensure_reference_assets(
    model_dir: Path,
    reference_audio: Optional[Path],
    reference_text: Optional[str],
) -> None:
    """Ensure the model directory contains a prompt WAV and metadata for reference audio."""
    if not reference_audio or not reference_audio.is_file():
        return

    target_path = model_dir / "prompt.wav"
    try:
        shutil.copy2(reference_audio, target_path)
    except Exception as exc:
        logging.warning("Failed to copy reference audio to %s: %s", target_path, exc)
        return

    prompt_text = reference_text or target_path.stem
    prompt_json = model_dir / "prompt_wav.json"
    prompt_data: dict = {}

    if prompt_json.is_file():
        try:
            prompt_data = json.loads(prompt_json.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Failed to parse existing prompt_wav.json in %s: %s", model_dir, exc)
            prompt_data = {}

    if not isinstance(prompt_data, dict):
        prompt_data = {}

    prompt_data["Normal"] = {
        "wav": target_path.name,
        "text": prompt_text,
    }

    try:
        prompt_json.write_text(json.dumps(prompt_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logging.warning("Failed to write prompt_wav.json in %s: %s", model_dir, exc)


def discover_prebuilt_assets(character_dir: Path) -> Optional[CharacterAssets]:
    """Discover assets from an existing Genie ONNX bundle (no conversion needed)."""
    if not has_onnx_bundle(character_dir):
        return None

    reference_audio: Optional[Path] = None
    reference_text: Optional[str] = None

    prompt_json = character_dir / "prompt_wav.json"
    if prompt_json.is_file():
        try:
            prompt_data = json.loads(prompt_json.read_text(encoding="utf-8"))
            preferred_order = ["Normal"]
            selected_entry = None

            if isinstance(prompt_data, dict):
                for key in preferred_order:
                    if key in prompt_data:
                        selected_entry = prompt_data[key]
                        break
                if selected_entry is None:
                    # Pick the first valid entry
                    for entry in prompt_data.values():
                        if isinstance(entry, dict) and ("wav" in entry or "text" in entry):
                            selected_entry = entry
                            break

            if isinstance(selected_entry, dict):
                wav_name = selected_entry.get("wav")
                if wav_name:
                    candidate = character_dir / wav_name
                    if candidate.is_file():
                        reference_audio = candidate.resolve()
                reference_text = selected_entry.get("text")
        except Exception as exc:
            logging.warning("Failed to parse prompt_wav.json in %s: %s", character_dir, exc)

    if reference_audio is None:
        fallback_wav = character_dir / "prompt.wav"
        if fallback_wav.is_file():
            reference_audio = fallback_wav.resolve()

    if reference_audio is None:
        wav_candidates = sorted(character_dir.glob("*.wav"))
        if wav_candidates:
            reference_audio = wav_candidates[0].resolve()

    if reference_audio is None:
        logging.warning("No reference audio found for prebuilt character at %s", character_dir)
        return None

    if not reference_text:
        reference_text = reference_audio.stem

    return CharacterAssets(
        model_dir=character_dir,
        reference_audio=reference_audio,
        reference_text=reference_text,
        prebuilt=True,
    )


def convert_to_onnx_if_needed(
    model_dir: Path,
    ckpt_path: Optional[Path],
    pth_path: Optional[Path],
    reference_audio: Optional[Path] = None,
    reference_text: Optional[str] = None,
) -> None:
    """Convert ckpt/pth weights to an ONNX bundle if the target directory is empty."""
    if has_onnx_bundle(model_dir):
        logging.info("Existing ONNX bundle detected in %s", model_dir)
        ensure_reference_assets(model_dir, reference_audio, reference_text)
        return

    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing T2S checkpoint: {ckpt_path}")
    if pth_path is None or not pth_path.is_file():
        raise FileNotFoundError(f"Missing VITS weights: {pth_path}")

    logging.info("ONNX bundle not found. Converting from ckpt/pth ...")
    model_dir.mkdir(parents=True, exist_ok=True)
    genie.convert_to_onnx(
        torch_pth_path=str(pth_path),
        torch_ckpt_path=str(ckpt_path),
        output_dir=str(model_dir),
    )
    logging.info("Conversion completed; ONNX files placed under %s", model_dir)
    ensure_reference_assets(model_dir, reference_audio, reference_text)


def load_character(display_name: str, character_key: str, model_dir: Path) -> None:
    """Load the ONNX bundle into genie_tts."""
    logging.info("Loading character '%s' from %s", display_name, model_dir)
    success = genie.load_character(
        character_name=character_key,
        onnx_model_dir=str(model_dir),
    )
    if success is False:
        raise RuntimeError(f"Failed to load character '{display_name}' from {model_dir}")


def configure_reference_audio(
    display_name: str,
    character_key: str,
    audio_path: Path,
    audio_text: str,
) -> None:
    """Set the reference audio so the voice style matches the training speaker."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")

    logging.info("Configuring reference audio for '%s': %s", display_name, audio_path)
    genie.set_reference_audio(
        character_name=character_key,
        audio_path=str(audio_path),
        audio_text=audio_text,
    )


def calculate_wav_duration(path: Path) -> float:
    """Return the duration (seconds) of a WAV file."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        framerate = wf.getframerate()
        if framerate == 0:
            return 0.0
        return frames / float(framerate)


def synthesize_line(session: CharacterSession, text: str, output_path: Path) -> float:
    """Generate a single line, store it as a WAV file, and return its duration."""
    configure_reference_audio(
        session.display_name,
        session.key,
        session.assets.reference_audio,
        session.assets.reference_text,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Generating speech for '%s' -> %s", session.display_name, output_path)
    genie.tts(
        character_name=session.key,
        text=text,
        play=False,
        save_path=str(output_path)
    )
    duration = calculate_wav_duration(output_path)
    logging.info("Generation completed (%.2f s).", duration)
    return duration


def review_generated_audio(generated: list[tuple[Path, float]]) -> None:
    """Play generated audio files sequentially using PyAudio."""
    if not generated:
        return

    try:
        import pyaudio
    except ImportError:
        logging.warning("Cannot review audio because 'pyaudio' is not installed.")
        return

    pa = pyaudio.PyAudio()
    chunk_size = 1024
    try:
        for audio_path, duration in generated:
            logging.info("Reviewing %s (%.2f s)", audio_path, duration)
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    stream = pa.open(
                        format=pa.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                    )
                    data = wf.readframes(chunk_size)
                    while data:
                        stream.write(data)
                        data = wf.readframes(chunk_size)
                    stream.stop_stream()
                    stream.close()
            except Exception as exc:
                logging.error("Failed to play %s: %s", audio_path, exc)
    finally:
        pa.terminate()


def select_latest_file(directory: Path, pattern: str) -> Path:
    """Pick the most recently updated file matching the pattern."""
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")
    return max(candidates, key=lambda file: file.stat().st_mtime)


def parse_slicer_list(list_path: Path, character_dir: Path) -> Optional[tuple[Path, str]]:
    """Parse the slicer list file to find an existing reference audio and text."""
    logging.info("Parsing slicer list: %s", list_path)
    lines = list_path.read_text(encoding="utf-8").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|")
        if len(parts) < 4:
            continue

        raw_rel_path, _, _, raw_text = parts[:4]
        rel_path = Path(raw_rel_path.strip().replace("\\", "/"))
        text = raw_text.strip()

        candidates = [
            list_path.parent / rel_path,
            list_path.parent / rel_path.name,
            character_dir / rel_path,
            character_dir / rel_path.name,
        ]

        for candidate in candidates:
            if candidate.is_file():
                logging.info("Matched reference entry via list file: %s", candidate)
                return candidate.resolve(), text

    logging.warning("No matching audio entries found in %s", list_path)
    return None


def discover_sovits_assets(character_dir: Path) -> CharacterAssets:
    """Gather SoVITS assets and reference metadata for the target character."""
    ckpt_path = select_latest_file(character_dir, "*.ckpt").resolve()
    pth_path = select_latest_file(character_dir, "*.pth").resolve()

    list_candidates = sorted(character_dir.glob("*.list"))
    reference_entry: Optional[tuple[Path, str]] = None
    for list_path in list_candidates:
        reference_entry = parse_slicer_list(list_path, character_dir)
        if reference_entry:
            break

    if reference_entry is None:
        wav_candidates = sorted(character_dir.glob("*.wav"))
        if not wav_candidates:
            raise FileNotFoundError(f"No reference audio files found in {character_dir}")
        logging.warning("Falling back to the first WAV file because no list entry matched.")
        reference_audio = wav_candidates[0].resolve()
        reference_text = reference_audio.stem
    else:
        reference_audio, reference_text = reference_entry

    return CharacterAssets(
        model_dir=GENIE_ROOT / character_dir.name,
        reference_audio=reference_audio,
        reference_text=reference_text,
        ckpt_path=ckpt_path,
        pth_path=pth_path,
        prebuilt=False,
    )


def resolve_sovits_directory(character_name: str) -> Path:
    """Resolve the SoVITS directory for a character (case-insensitive)."""
    if not SOVITS_ROOT.is_dir():
        raise FileNotFoundError(f"SoVITS root not found: {SOVITS_ROOT}")

    character_dir = find_character_dir(SOVITS_ROOT, character_name)
    if character_dir:
        return character_dir

    raise FileNotFoundError(
        f"Could not locate SoVITS directory for character '{character_name}' in {SOVITS_ROOT}"
    )


def prepare_character_session(
    requested_name: str,
    session_cache: dict[str, CharacterSession],
) -> CharacterSession:
    """Return a prepared character session, creating it if necessary."""
    assets: Optional[CharacterAssets] = None
    display_name: Optional[str] = None
    character_key: Optional[str] = None

    prebuilt_dir = find_character_dir(GENIE_ROOT, requested_name)
    if prebuilt_dir:
        display_name = prebuilt_dir.name
        character_key = display_name.lower()
        cached = session_cache.get(character_key)
        if cached:
            return cached

        assets = discover_prebuilt_assets(prebuilt_dir)
        if assets is None:
            logging.warning(
                "Prebuilt Genie model detected for '%s' but reference assets are incomplete; "
                "falling back to SoVITS conversion (if available).",
                display_name,
            )
        else:
            logging.info("Using prebuilt Genie model for character '%s'.", display_name)

    if assets is None:
        character_dir = resolve_sovits_directory(requested_name)
        display_name = character_dir.name
        character_key = display_name.lower()
        cached = session_cache.get(character_key)
        if cached:
            return cached

        assets = discover_sovits_assets(character_dir)

    assert assets is not None
    assert display_name is not None
    assert character_key is not None

    model_dir = assets.model_dir
    if not assets.prebuilt:
        ensure_directories([model_dir])
        convert_to_onnx_if_needed(
            model_dir,
            assets.ckpt_path,
            assets.pth_path,
            assets.reference_audio,
            assets.reference_text,
        )
    else:
        ensure_reference_assets(model_dir, assets.reference_audio, assets.reference_text)

    load_character(display_name, character_key, model_dir)
    configure_reference_audio(display_name, character_key, assets.reference_audio, assets.reference_text)

    session = CharacterSession(
        display_name=display_name,
        key=character_key,
        model_dir=model_dir,
        assets=assets,
    )
    session_cache[character_key] = session
    return session


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the CSV file and return cleaned rows."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file '{csv_path}' must contain headers (expected 'Chara,Dialog').")

        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Batch-generate speech lines from a CSV file."""
    setup_logging()

    if not CSV_PATH.is_file():
        logging.error("CSV file not found: %s", CSV_PATH)
        return
    if not SOVITS_ROOT.is_dir():
        logging.warning("SoVITS root not found (%s). Only prebuilt Genie models will be available.", SOVITS_ROOT)

    script_output_root = OUTPUT_ROOT / SCRIPT_NAME
    ensure_directories([GENIE_ROOT, script_output_root])

    try:
        rows = read_csv_rows(CSV_PATH)
    except Exception as exc:
        logging.error("Failed to read CSV file '%s': %s", CSV_PATH, exc)
        return

    if not rows:
        logging.warning("CSV file '%s' does not contain any data rows.", CSV_PATH)
        return

    session_cache: dict[str, CharacterSession] = {}
    generated_outputs: list[tuple[Path, float]] = []
    overall_counter = 0
    successes = 0
    failures = 0

    for row_index, raw_row in enumerate(rows, start=1):
        normalized = {
            (key or "").strip().lower(): (value or "").strip()
            for key, value in raw_row.items()
        }

        character_name = normalized.get("chara")
        dialog_text = normalized.get("dialog")

        if not character_name:
            logging.warning("Row %d: missing 'Chara' value; skipping.", row_index)
            failures += 1
            continue
        if not dialog_text:
            logging.warning("Row %d: missing 'Dialog' value; skipping.", row_index)
            failures += 1
            continue

        overall_counter += 1
        output_filename = f"{overall_counter:03d}.wav"
        output_path = script_output_root / output_filename

        try:
            session = prepare_character_session(character_name, session_cache)
            session.line_counter += 1
            duration = synthesize_line(session, dialog_text, output_path)
            generated_outputs.append((output_path, duration))
            logging.info(
                "Saved line %03d (character '%s') -> %s",
                overall_counter,
                session.display_name,
                output_path,
            )
            successes += 1
        except Exception as exc:
            logging.exception(
                "Row %d: failed to synthesize line for character '%s': %s",
                row_index,
                character_name,
                exc,
            )
            logging.error(
                "Line %03d skipped due to error; reserved output path: %s",
                overall_counter,
                output_path,
            )
            failures += 1

    logging.info("Generation complete. %d succeeded, %d failed.", successes, failures)

    if REVIEW_GEN_AUDIO and generated_outputs:
        review_generated_audio(generated_outputs)


if __name__ == "__main__":
    main()
