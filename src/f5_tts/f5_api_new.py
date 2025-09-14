import random
import sys
from importlib.resources import files
import tempfile
import torchaudio
import soundfile as sf
from cached_path import cached_path
import argparse
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything
import torch
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat
from f5_tts.utils.whisper_api import translate_inference,transribe_inference

#ถ้าอยากใช้โมเดลที่อัพเดทใหม หรือโมเดลภาษาอื่น สามารถแก้ไขโค้ด Model และ Vocab เช่น default_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_350000.pt"
default_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt"
v2_model_base = "hf://VIZINTZOR/F5-TTS-TH-v2/model_250000.pt"
vocab_base = "./vocab/vocab.txt"
vocab_ipa_base = "./vocab/vocab_ipa.txt"

model_choices = ["Default", "V2", "Custom"]

global f5tts_model
f5tts_model = None

def load_f5tts(ckpt_path, vocab_path=vocab_base, model_type="v1"):
    if model_type == "v1":
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)
    elif model_type == "v2":
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=True, conv_layers=4, pe_attn_head=None)
        vocab_path = "./vocab/vocab_ipa.txt"
    model = load_model(DiT, F5TTS_model_cfg, ckpt_path, vocab_file = vocab_path, use_ema=True)
    print(f"Loaded model from {ckpt_path}")
    return model

vocoder = load_vocoder()

f5tts_model = load_f5tts(str(cached_path(default_model_base)))

def load_custom_model(model_choice,model_custom_path):
    torch.cuda.empty_cache()
    global f5tts_model
    model_path = default_model_base if model_choice == "Default" else v2_model_base
    if model_choice == "Custom":
        f5tts_model = load_f5tts(str(cached_path(model_custom_path)))
        return f"Loaded Custom Model {model_custom_path}"
    else:
        f5tts_model = load_f5tts(
            str(cached_path(model_path)),
            vocab_path = vocab_ipa_base if model_choice == "V2" else vocab_base,
            model_type = "v2" if model_choice == "V2" else "v1"
        )
        return f"Loaded Model {model_choice}"
    
def infer_tts(
    ref_audio_orig,
    ref_text,
    gen_text,
    remove_silence=True,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    cfg_strength=2,
    max_chars=250,
    seed=-1,
    lang_process="Default",
):
    """Run inference and return (sample_rate, numpy_wave), spectrogram_path, ref_text, seed"""
    global f5tts_model
    if f5tts_model is None:
        f5tts_model = load_f5tts(str(cached_path(default_model_base)))

    if seed == -1:
        seed = random.randint(0, sys.maxsize)
    seed_everything(seed)
    output_seed = seed

    if not ref_audio_orig:
        raise ValueError("Please provide reference audio.")

    if not gen_text.strip():
        raise ValueError("Please enter text to generate.")

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)

    gen_text_cleaned = process_thai_repeat(replace_numbers_with_thai(gen_text))

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text_cleaned,
        f5tts_model,
        vocoder,
        cross_fade_duration=float(cross_fade_duration),
        nfe_step=nfe_step,
        speed=speed,
        progress=None,
        cfg_strength=cfg_strength,
        set_max_chars=max_chars,
        use_ipa=True if lang_process == "IPA" else False,
    )

    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    print("seed:", output_seed)
    return (final_sample_rate, final_wave), spectrogram_path, ref_text, output_seed

def transcribe_text(input_audio="",translate=False,model="large-v3-turbo",compute_type="float16",target_lg="th",source_lg='th'):
    if translate:
        output_text = translate_inference(text=transribe_inference(input_audio=input_audio,model=model,
                                          compute_type=compute_type,language=source_lg),target=target_lg)
    else:
        output_text = transribe_inference(input_audio=input_audio,model=model,
                                          compute_type=compute_type,language=source_lg)
    return output_text

app = FastAPI(title="F5-TTS API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
async def infer_endpoint(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    gen_text: str = Form(...),
    remove_silence: bool = Form(True),
    cross_fade_duration: float = Form(0.15),
    nfe_step: int = Form(32),
    speed: float = Form(1.0),
    cfg_strength: float = Form(2.0),
    max_chars: int = Form(250),
    seed: int = Form(-1),
    lang_process: str = Form("Default"),
    model_choice: str = Form("Default"),
    model_custom: str = Form(""),
):
    # validate
    if ref_audio.content_type.split("/")[0] != "audio":
        raise HTTPException(status_code=400, detail="ref_audio must be an audio file")

    # optionally load model
    if model_choice in model_choices:
        load_custom_model(model_choice, model_custom)

    # save upload to temp file
    import tempfile

    try:
        suffix = os.path.splitext(ref_audio.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await ref_audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # call infer_tts
        (sr, wave), spect_path, out_ref_text, out_seed = infer_tts(
            tmp_path,
            ref_text,
            gen_text,
            remove_silence=remove_silence,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            cfg_strength=cfg_strength,
            max_chars=max_chars,
            seed=seed,
            lang_process=lang_process,
        )

        # stream back wav
        buf = io.BytesIO()
        sf.write(buf, wave, sr, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def main():
    import uvicorn
    uvicorn.run("src.f5_tts.f5_api_new:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()



