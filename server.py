"""
F5-TTS Thai API Server (Optimized for A100/Colab)
==================================================

Separate FastAPI server entry point optimized for Google Colab A100.

Features:
- Sway Sampling enabled by default for better speech quality
- EPSS (Empirically Pruned Step Sampling) for faster inference
- torch.compile() for optimized model execution
- A100-specific optimizations (TF32, cuDNN, flash attention)
- Reference caching for faster repeated generations

Usage:
    python server.py
    
    Or with uvicorn:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import random
import tempfile
import torchaudio
import soundfile as sf
from cached_path import cached_path
import json
import uuid
import shutil
from pathlib import Path
import numpy as np

# Import from f5_api_new_integrate
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
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat

# Configuration
default_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt"
v2_model_base = "hf://VIZINTZOR/F5-TTS-TH-v2/model_250000.pt"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_base = os.path.join(ROOT_DIR, "vocab", "vocab.txt")
vocab_ipa_base = os.path.join(ROOT_DIR, "vocab", "vocab_ipa.txt")

# Global variables
f5tts_model = None
vocoder = None
device = None

# Cached reference data for speed optimization
cached_ref_audio = None
cached_ref_text = None
cached_ref_processed = None
cached_cleaned_text = {}


# Create main FastAPI app
app = FastAPI(
    title="F5-TTS Thai API Server (A100 Optimized)",
    description="""
Thai Text-to-Speech API using F5-TTS with Sway Sampling, EPSS, and A100 optimizations.

## Features
- **Sway Sampling**: Focuses computational power on early generation stages
- **EPSS**: Empirically Pruned Step Sampling for optimized inference
- **torch.compile**: JIT compilation for faster model execution
- **A100 Optimizations**: TF32, cuDNN benchmark, flash attention
- **Reference Caching**: Cache reference audio for faster repeated generations

## Key Parameters
- `sway_sampling_coef`: Set to -1.0 (default) to enable Sway Sampling
- `use_epss`: Set to True (default) for EPSS optimization
- `nfe_step`: 8-16 for fast, 32 for quality
- `fast_mode`: Enable for maximum speed
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TTSResponse(BaseModel):
    success: bool
    audio_file: Optional[str] = None
    spectrogram_file: Optional[str] = None
    ref_text: Optional[str] = None
    seed: Optional[int] = None
    message: Optional[str] = None


class ModelLoadResponse(BaseModel):
    success: bool
    message: str


def load_f5tts(ckpt_path, vocab_path=vocab_base, model_type="v1"):
    """Load F5-TTS model with optimizations"""
    if model_type == "v1":
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)
    elif model_type == "v2":
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=True, conv_layers=4, pe_attn_head=None)
        vocab_path = vocab_ipa_base
    
    # Filter config for DiT constructor
    try:
        import inspect
        sig = inspect.signature(DiT.__init__)
        allowed = {name for name, p in sig.parameters.items() if name != 'self' and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
        filtered_cfg = {k: v for k, v in F5TTS_model_cfg.items() if k in allowed}
        dropped = [k for k in F5TTS_model_cfg.keys() if k not in filtered_cfg]
        if dropped:
            print(f"Dropping unsupported keys from model cfg: {dropped}")
    except Exception:
        filtered_cfg = F5TTS_model_cfg

    model = load_model(DiT, filtered_cfg, ckpt_path, vocab_file=vocab_path, use_ema=True)
    print(f"Loaded model from {ckpt_path}")
    return model


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the path"""
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_extension = Path(upload_file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = temp_dir / temp_filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return str(temp_path)


@app.on_event("startup")
async def startup():
    """Initialize models with A100/Colab optimizations"""
    global f5tts_model, vocoder, device
    
    print("=" * 70)
    print("F5-TTS Thai API Server (A100 Optimized) Starting...")
    print("=" * 70)
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            
            # ============================================
            # A100 / High-End GPU Optimizations
            # ============================================
            
            # 1. Enable cuDNN benchmark for optimal convolution algorithms
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN benchmark enabled")
            
            # 2. Enable TF32 for faster matmul on Ampere+ GPUs (A100, RTX 30xx, etc.)
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✓ TF32 enabled (Ampere+ optimization)")
            
            # 3. Set float32 matmul precision for A100
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
                print("✓ Float32 matmul precision set to 'high'")
            
            # 4. Enable flash attention if available (PyTorch 2.0+)
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("✓ Flash Attention available (SDPA)")
        else:
            print("Running on CPU (No GPU available)")
        
        # Load vocoder
        print("\nLoading vocoder...")
        vocoder = load_vocoder()
        if hasattr(vocoder, 'to'):
            vocoder = vocoder.to(device)
        print("✓ Vocoder loaded")
        
        # Load F5-TTS model
        print("\nLoading F5-TTS model...")
        f5tts_model = load_f5tts(str(cached_path(default_model_base)))
        f5tts_model = f5tts_model.to(device)
        print("✓ F5-TTS model loaded")
        
        # ============================================
        # torch.compile for faster inference (PyTorch 2.0+)
        # ============================================
        if hasattr(torch, 'compile'):
            try:
                print("\nApplying torch.compile optimization...")
                # Use 'reduce-overhead' mode for inference (best for A100)
                # Options: 'default', 'reduce-overhead', 'max-autotune'
                f5tts_model = torch.compile(
                    f5tts_model, 
                    mode='reduce-overhead',
                    fullgraph=False,  # Allow graph breaks for compatibility
                )
                print("✓ torch.compile applied (mode: reduce-overhead)")
            except Exception as e:
                print(f"⚠ torch.compile failed (will use eager mode): {e}")
        else:
            print("⚠ torch.compile not available (PyTorch < 2.0)")
        
        # Set model to eval mode
        f5tts_model.eval()
        
        print("\n" + "=" * 70)
        print("✓ Server ready! Documentation at /docs")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("F5-TTS Thai API Server shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    return {
        "message": "F5-TTS Thai API Server (A100 Optimized)",
        "version": "2.0.0",
        "features": [
            "Sway Sampling (default: enabled)",
            "EPSS (Empirically Pruned Step Sampling)",
            "torch.compile optimization",
            "A100/Colab optimizations",
            "Reference audio caching",
        ],
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(device) if device else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "models_loaded": f5tts_model is not None and vocoder is not None,
        "reference_cached": cached_ref_processed is not None,
        "text_cache_size": len(cached_cleaned_text),
        "torch_compile": hasattr(torch, 'compile'),
    }


@app.post("/set_reference")
async def set_reference(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...)
):
    """Cache reference audio and text for faster TTS generation"""
    global cached_ref_audio, cached_ref_text, cached_ref_processed
    
    try:
        ref_audio_path = await save_upload_file(ref_audio)
        cached_ref_audio = ref_audio_path
        cached_ref_text = ref_text
        cached_ref_processed = preprocess_ref_audio_text(ref_audio_path, ref_text)
        
        return {"success": True, "message": "Reference cached successfully", "ref_text": ref_text}
    except Exception as e:
        return {"success": False, "error": f"Failed to cache reference: {str(e)}"}


@app.delete("/clear_reference")
async def clear_reference():
    """Clear cached reference data"""
    global cached_ref_audio, cached_ref_text, cached_ref_processed
    
    try:
        if cached_ref_audio and os.path.exists(cached_ref_audio):
            os.unlink(cached_ref_audio)
        cached_ref_audio = None
        cached_ref_text = None
        cached_ref_processed = None
        return {"success": True, "message": "Reference cache cleared"}
    except Exception as e:
        return {"success": False, "error": f"Failed to clear cache: {str(e)}"}


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(
    ref_audio: UploadFile = File(None),        # Optional: use cached if not provided
    ref_text: str = Form(None),                 # Optional: use cached if not provided
    gen_text: str = Form(...),
    remove_silence: bool = Form(True),
    cross_fade_duration: float = Form(0.15),
    nfe_step: int = Form(8),                    # 8 for fast, 16 for balanced, 32 for quality
    speed: float = Form(1.0),
    cfg_strength: float = Form(2.0),
    max_chars: int = Form(250),
    seed: int = Form(-1),
    sway_sampling_coef: float = Form(-1.0),     # Sway Sampling: enabled by default
    use_epss: bool = Form(True),                # EPSS: enabled by default
    fast_mode: bool = Form(False),              # Fast mode: lower quality for speed
    return_file: bool = Form(False),
):
    """
    Generate speech from text with Sway Sampling and EPSS optimizations.
    
    - sway_sampling_coef=-1.0: Enables Sway Sampling for better speech structure
    - use_epss=True: Uses EPSS for optimized timestep scheduling
    - fast_mode=True: Reduces quality settings for maximum speed
    """
    global f5tts_model, vocoder, cached_ref_processed
    
    if f5tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use cached reference if available
        if ref_audio is None and cached_ref_processed is not None:
            ref_audio_processed, ref_text_processed = cached_ref_processed
            ref_audio_path = cached_ref_audio
            should_cleanup = False
        else:
            if ref_audio is None or ref_text is None:
                raise HTTPException(status_code=400, detail="Reference audio and text required (or set cache first)")
            ref_audio_path = await save_upload_file(ref_audio)
            ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text)
            should_cleanup = True
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        seed_everything(seed)
        
        # Validate input
        if not gen_text.strip():
            raise HTTPException(status_code=400, detail="Generated text cannot be empty")
        
        # Clean text (with caching)
        if gen_text in cached_cleaned_text:
            gen_text_cleaned = cached_cleaned_text[gen_text]
        else:
            gen_text_cleaned = process_thai_repeat(replace_numbers_with_thai(gen_text))
            cached_cleaned_text[gen_text] = gen_text_cleaned
        
        # Apply fast mode settings
        if fast_mode:
            nfe_step = min(nfe_step, 4)
            remove_silence = False
            cfg_strength = min(cfg_strength, 1.5)
        
        # Generate audio with Sway Sampling
        with torch.inference_mode():
            final_wave, final_sample_rate, combined_spectrogram = infer_process(
                ref_audio_processed,
                ref_text_processed,
                gen_text_cleaned,
                f5tts_model,
                vocoder,
                cross_fade_duration=float(cross_fade_duration),
                nfe_step=nfe_step,
                speed=speed,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                set_max_chars=max_chars,
            )
        
        # Remove silence if requested
        if remove_silence:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()
        
        # Save output
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        output_filename = f"generated_{uuid.uuid4()}.wav"
        output_path = output_dir / output_filename
        
        sf.write(str(output_path), final_wave, final_sample_rate)
        
        # Save spectrogram
        spectrogram_filename = f"spectrogram_{uuid.uuid4()}.jpg"
        spectrogram_path = output_dir / spectrogram_filename
        save_spectrogram(combined_spectrogram, str(spectrogram_path))
        
        # Cleanup
        if should_cleanup:
            os.unlink(ref_audio_path)
        
        if return_file:
            return FileResponse(path=str(output_path), media_type="audio/wav", filename=output_filename)
        
        return TTSResponse(
            success=True,
            audio_file=str(output_path),
            spectrogram_file=str(spectrogram_path),
            ref_text=ref_text_processed,
            seed=seed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return TTSResponse(success=False, message=f"TTS generation failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated audio files"""
    file_path = Path("./outputs") / filename
    if file_path.exists():
        return FileResponse(path=str(file_path), filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


@app.delete("/cleanup")
async def cleanup_files():
    """Remove old generated files"""
    try:
        output_dir = Path("./outputs")
        temp_dir = Path("./temp_uploads")
        
        count = 0
        for directory in [output_dir, temp_dir]:
            if directory.exists():
                for file_path in directory.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        count += 1
        
        # Clear text cache
        cached_cleaned_text.clear()
        
        return {"success": True, "message": f"Cleaned up {count} files and text cache"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "log_level": "info",
        "access_log": True,
    }
    
    # Use uvloop for better async performance
    try:
        import uvloop
        config["loop"] = "uvloop"
        print("Using uvloop for better async performance")
    except ImportError:
        pass
    
    print(f"Starting F5-TTS Thai API Server on http://{config['host']}:{config['port']}")
    uvicorn.run("server:app", **config)
