import random
import re
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000

# 'a' = американский английский, 'b' = британский
pipelines = {}


def get_pipeline(lang_code):
    if lang_code not in pipelines:
        pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return pipelines[lang_code]


VOICES = {
    "Michael (US, мужской, документальный)": ("am_michael", "a"),
    "Adam (US, мужской, глубокий)": ("am_adam", "a"),
    "Fenrir (US, мужской)": ("am_fenrir", "a"),
    "Michael + Adam (микс, если не запустится - выбери другой)": ("am_michael,am_adam", "a"),
    "George (UK, мужской)": ("bm_george", "b"),
    "Heart (US, женский)": ("af_heart", "a"),
    "Bella (US, женский)": ("af_bella", "a"),
}

# Теги из сценария теперь работают: паузы и изменение темпа
PAUSE_TAGS = {
    "short pause": 0.45,
    "pause": 0.6,
    "long pause": 1.0,
    "sighs": 0.5,
    "sigh": 0.5,
    "beat": 0.3,
}
STYLE_TAGS = {
    "serious": 0.93,
    "calm": 0.90,
    "thoughtful": 0.92,
    "dramatic": 0.90,
    "excited": 1.05,
    "surprised": 1.03,
}


def split_sentences(text):
    parts = re.split(r'(?<=\.\.\.)\s+|(?<=[.!?…])\s+(?=["“0-9A-Z])', text)
    return [p.strip() for p in parts if p.strip()]


def parse(text):
    events = []
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


def end_pause(sentence, rng):
    s = sentence.rstrip('"”\' ')
    if s.endswith("...") or s.endswith("…"):
        base = 0.55
    elif s.endswith("?"):
        base = 0.45
    elif s.endswith("!"):
        base = 0.35
    elif s.endswith(":"):
        base = 0.40
    elif s.endswith("."):
        base = 0.38
    else:
        base = 0.30
    return base * (1 + rng.uniform(-0.15, 0.15))


def trim_silence(a, rel_thr=0.02, margin_sec=0.03):
    peak = np.max(np.abs(a)) if len(a) else 0
    if peak < 1e-4:
        return a
    idx = np.where(np.abs(a) > rel_thr * peak)[0]
    margin = int(margin_sec * SAMPLE_RATE)
    start = max(idx[0] - margin, 0)
    end = min(idx[-1] + margin, len(a))
    return a[start:end]


def fade(a, sec=0.008):
    n = min(int(sec * SAMPLE_RATE), len(a) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        a = a.copy()
        a[:n] *= ramp
        a[-n:] *= ramp[::-1]
    return a


def silence(sec):
    return np.zeros(int(SAMPLE_RATE * sec), dtype=np.float32)


def synthesize(text, voice_name, speed, para_pause, variation, seed, progress=gr.Progress()):
    text = text.strip()
    if not text:
        return None

    voice, lang = VOICES[voice_name]
    pipeline = get_pipeline(lang)
    rng = random.Random(int(seed))

    events = parse(text)
    total = sum(1 for e in events if e[0] == "say")
    done = 0

    chunks = []
    pending = 0.0

    for kind, value, style in events:
        if kind == "pause":
            pending = max(pending, value)
        elif kind == "para":
            pending = max(pending, para_pause)
        else:
            if chunks and pending > 0:
                chunks.append(silence(pending))
            pending = 0.0

            sent_speed = speed * style * (1 + rng.uniform(-variation, variation))
            parts = []
            for _, _, audio in pipeline(value, voice=voice, speed=sent_speed):
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                parts.append(np.asarray(audio, dtype=np.float32))
            if parts:
                a = np.concatenate(parts)
                a = fade(trim_silence(a))
                chunks.append(a)

            pending = end_pause(value, rng)
            done += 1
            progress(done / max(total, 1), desc=f"Предложение {done} из {total}")

    if not chunks:
        return None

    result = np.concatenate(chunks)
    peak = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * 0.9

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(out.name, result, SAMPLE_RATE)
    return out.name


with gr.Blocks(title="Моя озвучка v2") as demo:
    gr.Markdown("## Озвучка сценария v2 (Kokoro, без лимитов)")
    text = gr.Textbox(
        label="Сценарий. Теги [short pause], [long pause], [serious], [calm], [thoughtful] работают",
        lines=18,
    )
    with gr.Row():
        voice = gr.Dropdown(list(VOICES.keys()), value=list(VOICES.keys())[0], label="Голос")
        speed = gr.Slider(0.7, 1.3, value=0.95, step=0.05, label="Базовая скорость")
    with gr.Row():
        pause = gr.Slider(0.2, 1.5, value=0.6, step=0.1, label="Пауза между абзацами (сек)")
        variation = gr.Slider(0.0, 0.08, value=0.03, step=0.01, label="Вариация темпа (живость)")
        seed = gr.Number(value=7, label="Seed (поменяй, чтобы получить другую подачу)", precision=0)
    btn = gr.Button("Озвучить", variant="primary")
    audio_out = gr.Audio(label="Результат (скачай через три точки справа)", type="filepath")
    btn.click(synthesize, [text, voice, speed, pause, variation, seed], audio_out)

if __name__ == "__main__":
    demo.launch()
