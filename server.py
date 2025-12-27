"""
F5-TTS Thai API Server
======================

Separate FastAPI server entry point with optimizations for F5-TTS Thai.

Features:
- Sway Sampling enabled by default for better speech quality
- EPSS (Empirically Pruned Step Sampling) for faster inference
- CORS enabled for cross-origin requests
- Performance optimizations (cuDNN benchmark, torch.compile)

Usage:
    python server.py
    
    Or with uvicorn:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

# Import the API module
from f5_tts.f5_api_new import (
    app as f5_app,
    startup_event as f5_startup,
    f5tts_model,
    vocoder,
)

# Create main FastAPI app
app = FastAPI(
    title="F5-TTS Thai API Server",
    description="""
Thai Text-to-Speech API using F5-TTS with Sway Sampling and EPSS.

## Features
- **Sway Sampling**: Focuses computational power on early generation stages for better speech structure
- **EPSS**: Empirically Pruned Step Sampling for optimized inference speed
- **Reference Caching**: Cache reference audio for faster repeated generations
- **Fast Mode**: Trade quality for speed when needed

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


@app.on_event("startup")
async def startup():
    """Initialize models and optimizations on startup"""
    print("=" * 60)
    print("F5-TTS Thai API Server Starting...")
    print("=" * 60)
    
    # Enable performance optimizations
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled")
        
        # Enable TF32 for faster matmul on Ampere+ GPUs
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for faster computation")
    else:
        print("Running on CPU (GPU not available)")
    
    # Initialize F5-TTS models
    await f5_startup()
    
    print("=" * 60)
    print("Server ready! Documentation at /docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("F5-TTS Thai API Server shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Mount the F5-TTS API routes
app.mount("/api/v1", f5_app)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "F5-TTS Thai API Server",
        "version": "2.0.0",
        "features": [
            "Sway Sampling (default: enabled)",
            "EPSS (Empirically Pruned Step Sampling)",
            "Reference audio caching",
            "Fast mode option",
        ],
        "docs": "/docs",
        "api": "/api/v1",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "models_loaded": f5tts_model is not None and vocoder is not None,
    }


if __name__ == "__main__":
    import uvicorn
    
    # Server configuration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,  # Single worker for GPU models
        "log_level": "info",
        "access_log": True,
    }
    
    # Use uvloop for better async performance if available
    try:
        import uvloop
        config["loop"] = "uvloop"
        print("Using uvloop for better async performance")
    except ImportError:
        pass
    
    print(f"Starting F5-TTS Thai API Server on http://{config['host']}:{config['port']}")
    uvicorn.run("server:app", **config)
