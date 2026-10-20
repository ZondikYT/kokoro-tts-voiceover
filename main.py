import random
import re
import tempfile
import os
import logging
import gradio as gr
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Setup professional logging environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SAMPLE_RATE = 24000
pipelines = {}

def get_pipeline(lang_code):
    """Lazy-loads and caches neural TTS pipelines to save RAM."""
    if lang_code not in pipelines:
        logging.info(f"Initializing new KPipeline for language code: '{lang_code}'")
        pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return pipelines[lang_code]

# Voice maps matching name to model file and regional language pipeline
VOICES = {
    "Michael (US, Male, Documentary)": ("am_michael", "a"),
    "Adam (US, Male, Deep)": ("am_adam", "a"),
    "Fenrir (US, Male)": ("am_fenrir", "a"),
    "George (UK, Male)": ("bm_george", "b"),
    "Heart (US, Female)": ("af_heart", "a"),
    "Bella (US, Female)": ("af_bella", "a"),
}

PAUSE_TAGS = {"short pause": 0.45, "pause": 0.6, "long pause": 1.0, "sighs": 0.5, "sigh": 0.5, "beat": 0.3}
STYLE_TAGS = {"serious": 0.93, "calm": 0.90, "thoughtful": 0.92, "dramatic": 0.90, "excited": 1.05, "surprised": 1.03}

def split_sentences(text):
    """Splits paragraphs into clean structural sentences based on punctuation."""
    parts = re.split(r'(?<=\.\.\.)\s+|(?<=[.!?…])\s+(?=["“0-9A-Z])', text)
    return [p.strip() for p in parts if p.strip()]

def parse_scenario(text):
    """Parses scripts into granular events while preventing regular expression injection."""
    events = []
    # Security sanitization: squeeze spaces to prevent structural parsing flaws
    text = re.sub(r' +', ' ', text)
    
    # Process text layout blocks split by dual line breaks
    for para in re.split(r"\n\s*\n", text):
        style = 1.0
        for piece in re.split(r"(\[[^\]]*\])", para):
            if not piece.strip():
                continue
            if piece.startswith("[") and piece.endswith("]"):
                tag = piece[1:-1].strip().lower()
                if tag in PAUSE_TAGS:
                    events.append(("pause", PAUSE_TAGS[tag], None))
                elif tag in STYLE_TAGS:
                    style = STYLE_TAGS[tag]
                continue
            piece = " ".join(piece.split())
            for s in split_sentences(piece):
                events.append(("say", s, style))
        events.append(("para", None, None))
    return events

def calculate_end_pause(sentence, rng):
    """Injects dynamic punctuation-driven human pauses using uniform distributions."""
    s = sentence.rstrip('"”\' ')
    if s.endswith("...") or s.endswith("…"): base = 0.55
    elif s.endswith("?"): base = 0.45
    elif s.endswith("!"): base = 0.35
    elif s.endswith(":"): base = 0.40
    elif s.endswith("."): base = 0.38
    else: base = 0.30
    return base * (1 + rng.uniform(-0.15, 0.15))

def trim_silence(audio_arr, rel_thr=0.02, margin_sec=0.03):
    """Removes dead air spikes at sample edges based on relative amplitude peaks."""
    if len(audio_arr) == 0: 
        return audio_arr
    peak = np.max(np.abs(audio_arr))
    if peak < 1e-4: 
        return audio_arr
    idx = np.where(np.abs(audio_arr) > rel_thr * peak)[0]
    if len(idx) == 0: 
        return audio_arr
    margin = int(margin_sec * SAMPLE_RATE)
    start = max(idx[0] - margin, 0)
    end = min(idx[-1] + margin, len(audio_arr))
    return audio_arr[start:end]

def apply_edge_fade(audio_arr, sec=0.008):
    """Fades boundaries linearly to prevent abrupt digital audio clicks."""
    n = min(int(sec * SAMPLE_RATE), len(audio_arr) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio_arr = audio_arr.copy()
        audio_arr[:n] *= ramp
        audio_arr[-n:] *= ramp[::-1]
    return audio_arr

def generate_silence(sec):
    """Returns a zeroed numpy float array of specified duration."""
    return np.zeros(int(SAMPLE_RATE * sec), dtype=np.float32)

def synthesize_audio(text, voice_name, speed, para_pause, variation, seed, progress=gr.Progress()):
    """Orchestrates pipeline execution, data cleanup, and WAV generation."""
    text = text.strip()
    if not text:
        return None
        
    # Input size enforcement (Mitigates ReDoS vector attacks or memory leaks)
    if len(text) > 25000:
        logging.warning("Input payload too long. Truncating text to 25000 characters limit.")
        text = text[:25000]

    try:
        voice, lang = VOICES[voice_name]
        pipeline = get_pipeline(lang)
        rng = random.Random(int(seed))

        events = parse_scenario(text)
        total_phrases = sum(1 for e in events if e[0] == "say")
        done_phrases = 0

        chunks = []
        pending_duration = 0.0

        for kind, value, style in events:
            if kind == "pause":
                pending_duration = max(pending_duration, value)
            elif kind == "para":
                pending_duration = max(pending_duration, para_pause)
            else:
                if chunks and pending_duration > 0:
                    chunks.append(generate_silence(pending_duration))
                pending_duration = 0.0

                # Compute runtime pace variance
                sent_speed = speed * style * (1 + rng.uniform(-variation, variation))
                parts = []
                
                # Execute generation within a stable generator loop
                for _, _, audio in pipeline(value, voice=voice, speed=sent_speed):
                    if hasattr(audio, "numpy"):
                        audio = audio.numpy()
                    parts.append(np.asarray(audio, dtype=np.float32))
                
                if parts:
                    a = np.concatenate(parts)
                    a = apply_edge_fade(trim_silence(a))
                    chunks.append(a)

                pending_duration = calculate_end_pause(value, rng)
                done_phrases += 1
                progress(done_phrases / max(total_phrases, 1), desc=f"Processing phrase {done_phrases}/{total_phrases}")

        if not chunks:
            return None

        # Build full file array and run clipping protection normalization
        result = np.concatenate(chunks)
        peak = np.max(np.abs(result))
        if peak > 0:
            result = result / peak * 0.9

        # Safely handle operating system files to prevent unlinked temporary leaks
        fd, file_path = tempfile.mkstemp(suffix=".wav")
        try:
            sf.write(file_path, result, SAMPLE_RATE)
        finally:
            os.close(fd)
            
        logging.info(f"Audio processing finished successfully. Created artifact: {file_path}")
        return file_path

    except Exception as err:
        logging.error(f"Critical exception raised during TTS synthesis: {err}")
        raise gr.Error(f"Engine failure during audio synthesis pipeline: {err}")

# Build UI Interface Layout block
with gr.Blocks(title="Kokoro TTS Scenario Engine v2") as demo:
    gr.Markdown("## 🎙️ Kokoro TTS Scenario Reader v2")
    text_input = gr.Textbox(
        label="Enter Scenario Text (Supports inline tags like [short pause], [long pause], [serious], [calm])",
        lines=12,
        placeholder="Example: [serious] Hello agent. [long pause] Are you ready for your next assignment?"
    )
    with gr.Row():
        voice_dropdown = gr.Dropdown(list(VOICES.keys()), value=list(VOICES.keys())[0], label="Voice Model")
        speed_slider = gr.Slider(0.7, 1.3, value=0.95, step=0.05, label="Base Speech Speed")
    with gr.Row():
        pause_slider = gr.Slider(0.2, 1.5, value=0.6, step=0.1, label="Paragraph Break Delay (sec)")
        var_slider = gr.Slider(0.0, 0.08, value=0.03, step=0.01, label="Tempo Randomization Variance")
        seed_num = gr.Number(value=7, label="Random Generator Seed Number", precision=0)
    
    submit_btn = gr.Button("Synthesize Script", variant="primary")
    audio_output = gr.Audio(label="Download Generated WAV Output File", type="filepath")
    
    submit_btn.click(
        synthesize_audio, 
        [text_input, voice_dropdown, speed_slider, pause_slider, var_slider, seed_num], 
        audio_output
    )

if __name__ == "__main__":
    demo.launch()
