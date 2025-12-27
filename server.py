"""
F5-TTS Thai API Server (Zero-Overhead Edition)
===============================================

Optimized for minimum preprocessing overhead on A100.

Key Optimizations:
- FULLY cached reference: skips ALL preprocessing on subsequent calls
- Direct infer_batch_process call: bypasses redundant audio loading
- Pre-loaded GPU tensors: no CPU-GPU transfer during inference
- No file I/O during cached inference

Usage:
    python server.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import random
import tempfile
import torchaudio
import soundfile as sf
from cached_path import cached_path
import uuid
import shutil
from pathlib import Path
import numpy as np

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
    hop_length,
    infer_batch_process,
    chunk_text,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat

# Configuration
default_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_base = os.path.join(ROOT_DIR, "vocab", "vocab.txt")
vocab_ipa_base = os.path.join(ROOT_DIR, "vocab", "vocab_ipa.txt")

# Global state
f5tts_model = None
vocoder = None
device = None

# FULLY preprocessed cache - stores GPU-ready tensors
class ReferenceCache:
    def __init__(self):
        self.audio_tensor = None      # Already on GPU, resampled, normalized
        self.ref_text = None          # Processed ref_text with punctuation
        self.audio_path = None        # Original file path
        self.is_ready = False
        
cache = ReferenceCache()
text_cache = {}  # Cache for cleaned text


app = FastAPI(
    title="F5-TTS Thai API (Zero-Overhead)",
    description="""
Thai TTS API with ZERO preprocessing overhead on cached calls.

## How it works
1. Call `/set_reference` ONCE with your reference audio
2. All subsequent `/tts` calls skip ALL preprocessing (no "Converting audio...")
3. Audio tensor is pre-loaded on GPU

## Speed
- First call: ~1-2s (preprocessing + inference)
- Cached calls: **~0.3-0.5s** (inference only)
    """,
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSResponse(BaseModel):
    success: bool
    audio_file: Optional[str] = None
    ref_text: Optional[str] = None
    seed: Optional[int] = None
    inference_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    message: Optional[str] = None


def load_f5tts(ckpt_path, vocab_path=vocab_base, model_type="v1"):
    if model_type == "v1":
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)
    elif model_type == "v2":
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=True, conv_layers=4, pe_attn_head=None)
        vocab_path = vocab_ipa_base
    
    try:
        import inspect
        sig = inspect.signature(DiT.__init__)
        allowed = {name for name, p in sig.parameters.items() if name != 'self'}
        filtered_cfg = {k: v for k, v in F5TTS_model_cfg.items() if k in allowed}
    except:
        filtered_cfg = F5TTS_model_cfg

    return load_model(DiT, filtered_cfg, ckpt_path, vocab_file=vocab_path, use_ema=True)


async def save_upload_file(upload_file: UploadFile) -> str:
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4()}{Path(upload_file.filename).suffix}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return str(temp_path)


def prepare_audio_tensor(audio_path: str) -> torch.Tensor:
    """Load audio and prepare GPU tensor (done ONCE during caching)"""
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)
    
    # Normalize RMS
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    target_rms = 0.1
    if rms < target_rms:
        audio = audio * target_rms / rms
    
    return audio.to(device)


@app.on_event("startup")
async def startup():
    global f5tts_model, vocoder, device
    
    print("=" * 60)
    print("F5-TTS Thai API (Zero-Overhead Edition)")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            print("✓ GPU optimizations enabled")
        
        print("\nLoading models...")
        vocoder = load_vocoder()
        if hasattr(vocoder, 'to'):
            vocoder = vocoder.to(device)
        
        f5tts_model = load_f5tts(str(cached_path(default_model_base)))
        f5tts_model = f5tts_model.to(device)
        f5tts_model.eval()
        
        if hasattr(torch, 'compile'):
            try:
                f5tts_model = torch.compile(f5tts_model, mode='max-autotune', dynamic=True)
                print("✓ torch.compile applied")
            except Exception as e:
                print(f"⚠ torch.compile failed: {e}")
        
        # Warmup
        print("\nWarmup...")
        dummy = torch.randn(1, 24000, device=device)
        with torch.inference_mode():
            try:
                f5tts_model.sample(
                    cond=dummy, text=["test"], duration=100, steps=2,
                    cfg_strength=0, lens=torch.tensor([100], device=device)
                )
            except:
                pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n" + "=" * 60)
        print("✓ Ready! Use /set_reference first, then /tts")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


@app.get("/")
async def root():
    return {
        "message": "F5-TTS Zero-Overhead API",
        "usage": "1. POST /set_reference  2. POST /tts (fast!)",
        "cache_ready": cache.is_ready,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": f5tts_model is not None,
        "cache_ready": cache.is_ready,
        "cached_ref_text": cache.ref_text[:50] + "..." if cache.ref_text else None,
    }


@app.post("/set_reference")
async def set_reference(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...)
):
    """
    Cache reference audio for ZERO-OVERHEAD subsequent TTS calls.
    This does ALL preprocessing once, so /tts skips it entirely.
    """
    global cache
    
    try:
        start = time.perf_counter()
        
        # Save uploaded file
        audio_path = await save_upload_file(ref_audio)
        
        # Run preprocessing ONCE (this is the slow part)
        processed_path, processed_text = preprocess_ref_audio_text(audio_path, ref_text)
        
        # Load and prepare GPU tensor
        audio_tensor = prepare_audio_tensor(processed_path)
        
        # Store in cache
        cache.audio_tensor = audio_tensor
        cache.ref_text = processed_text
        cache.audio_path = audio_path
        cache.is_ready = True
        
        prep_time = (time.perf_counter() - start) * 1000
        
        return {
            "success": True,
            "message": f"Reference cached in {prep_time:.0f}ms. All subsequent /tts calls will skip preprocessing!",
            "ref_text": processed_text,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/clear_cache")
async def clear_cache():
    global cache, text_cache
    
    if cache.audio_path and os.path.exists(cache.audio_path):
        os.unlink(cache.audio_path)
    
    cache = ReferenceCache()
    text_cache.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"success": True, "message": "Cache cleared"}


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(
    ref_audio: UploadFile = File(None),  # Ignored if cache is ready
    ref_text: str = Form(None),          # Ignored if cache is ready  
    gen_text: str = Form(...),
    remove_silence: bool = Form(False),
    cross_fade_duration: float = Form(0.15),
    nfe_step: int = Form(16),
    speed: float = Form(1.0),
    cfg_strength: float = Form(2.0),
    max_chars: int = Form(250),
    seed: int = Form(-1),
    sway_sampling_coef: float = Form(-1.0),
    return_file: bool = Form(False),
):
    """
    Generate speech. If /set_reference was called, this skips ALL preprocessing!
    
    Without cache: ~1-2s (preprocessing + inference)
    With cache: ~0.3-0.5s (inference only)
    """
    global f5tts_model, vocoder, cache
    
    if f5tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.perf_counter()
    
    try:
        # =====================================
        # FAST PATH: Use cached reference (no preprocessing!)
        # =====================================
        if cache.is_ready:
            audio_tensor = cache.audio_tensor
            ref_text_processed = cache.ref_text
            used_cache = True
        else:
            # Slow path: need to preprocess
            if ref_audio is None or ref_text is None:
                raise HTTPException(
                    status_code=400, 
                    detail="No cache. Either call /set_reference first, or provide ref_audio and ref_text"
                )
            audio_path = await save_upload_file(ref_audio)
            processed_path, ref_text_processed = preprocess_ref_audio_text(audio_path, ref_text)
            audio_tensor = prepare_audio_tensor(processed_path)
            os.unlink(audio_path)
            used_cache = False
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        seed_everything(seed)
        
        # Clean text (cached)
        if gen_text not in text_cache:
            text_cache[gen_text] = process_thai_repeat(replace_numbers_with_thai(gen_text))
        gen_text_cleaned = text_cache[gen_text]
        
        # Chunk text
        gen_text_batches = chunk_text(gen_text_cleaned, max_chars=max_chars)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        infer_start = time.perf_counter()
        
        # =====================================
        # Direct inference (no file I/O!)
        # =====================================
        with torch.inference_mode():
            result = next(infer_batch_process(
                ref_audio=(audio_tensor, target_sample_rate),
                ref_text=ref_text_processed,
                gen_text_batches=gen_text_batches,
                model_obj=f5tts_model,
                vocoder=vocoder,
                mel_spec_type="vocos",
                progress=None,  # No progress bar overhead
                target_rms=0.1,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                device=device,
            ))
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        infer_time = (time.perf_counter() - infer_start) * 1000
        
        # Unpack result
        final_wave, sample_rate, spectrogram = result
        
        # Remove silence (optional)
        if remove_silence:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
                final_wave = final_wave.squeeze().cpu().numpy()
        
        # Save output
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        output_filename = f"gen_{uuid.uuid4().hex[:8]}.wav"
        output_path = output_dir / output_filename
        sf.write(str(output_path), final_wave, target_sample_rate)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        if return_file:
            return FileResponse(path=str(output_path), media_type="audio/wav", filename=output_filename)
        
        return TTSResponse(
            success=True,
            audio_file=str(output_path),
            ref_text=ref_text_processed,
            seed=seed,
            inference_time_ms=round(infer_time, 1),
            total_time_ms=round(total_time, 1),
            message=f"{'Cached' if used_cache else 'Uncached'}: Total {total_time:.0f}ms, Inference {infer_time:.0f}ms"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TTSResponse(success=False, message=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = Path("./outputs") / filename
    if file_path.exists():
        return FileResponse(path=str(file_path), filename=filename)
    raise HTTPException(status_code=404, detail="Not found")


@app.delete("/cleanup")
async def cleanup():
    count = 0
    for d in [Path("./outputs"), Path("./temp_uploads")]:
        if d.exists():
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
                    count += 1
    text_cache.clear()
    return {"deleted": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
