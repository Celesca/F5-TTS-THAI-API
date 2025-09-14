import random
import sys
from importlib.resources import files
import io
import os

import soundfile as sf
import tqdm
from cached_path import cached_path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything


class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        # Set device
        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # Load models
        self.load_vocoder_model(vocoder_name, local_path=local_path, hf_cache_dir=hf_cache_dir)
        self.load_ema_model(
            model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema, hf_cache_dir=hf_cache_dir
        )

    def load_vocoder_model(self, vocoder_name, local_path=None, hf_cache_dir=None):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device, hf_cache_dir)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, hf_cache_dir=None):
        if model_type == "F5-TTS":
            if not ckpt_file:
                if mel_spec_type == "vocos":
                    ckpt_file = str(
                        cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                    )
                elif mel_spec_type == "bigvgan":
                    ckpt_file = str(
                        cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt", cache_dir=hf_cache_dir)
                    )
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(
                    cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                )
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
        set_max_chars=250,
        use_ipa=False,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
            set_max_chars=set_max_chars,
            use_ipa=use_ipa,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect


if __name__ == "__main__":
    # If run directly, start a small FastAPI app for POST-based inference
    app = FastAPI(title="F5-TTS API")

    # create a single F5TTS instance to reuse models
    f5tts = F5TTS()

    # Model defaults (same as webui)
    default_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt"
    v2_model_base = "hf://VIZINTZOR/F5-TTS-TH-v2/model_250000.pt"
    vocab_base = "./vocab/vocab.txt"
    vocab_ipa_base = "./vocab/vocab_ipa.txt"

    model_choices = ["Default", "V2", "Custom"]


    def load_model_choice(model_choice: str = "Default", custom_path: str = ""):
        """Load and replace the underlying model in the shared f5tts instance."""
        global f5tts
        # choose checkpoint and vocab
        if model_choice == "Default":
            ckpt = str(cached_path(default_model_base))
            vocab = vocab_base
            use_ipa = False
        elif model_choice == "V2":
            ckpt = str(cached_path(v2_model_base))
            vocab = vocab_ipa_base
            use_ipa = True
        elif model_choice == "Custom":
            if not custom_path:
                raise ValueError("Custom model selected but no model path provided")
            ckpt = str(cached_path(custom_path))
            vocab = vocab_base
            use_ipa = False
        else:
            raise ValueError("Unknown model choice")

        # Reinitialize model weights (keep existing configs)
        f5tts.load_ema_model(f5tts.ema_model.transformer.config._get_name() if hasattr(f5tts.ema_model, 'transformer') else 'F5-TTS', ckpt, f5tts.mel_spec_type, vocab, 'euler', True)
        return use_ipa


    @app.get("/health")
    def health():
        return {"status": "ok", "device": f5tts.device}


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
        use_ipa: bool = Form(False),
        model_choice: str = Form("Default"),
        model_custom_path: str = Form(""),
    ):
        """Accept multipart/form-data with a reference audio file and text fields, return generated WAV"""

        # Validate upload
        if ref_audio.content_type.split("/")[0] != "audio":
            raise HTTPException(status_code=400, detail="ref_audio must be an audio file")

        # Save uploaded file to a temporary file path
        import tempfile

        try:
            suffix = os.path.splitext(ref_audio.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await ref_audio.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Optionally load model choice
            if model_choice and model_choice in model_choices:
                try:
                    auto_ipa = load_model_choice(model_choice, model_custom_path)
                    # prefer explicit use_ipa param if provided, otherwise use model default
                    if not use_ipa:
                        use_ipa = auto_ipa
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")

            # Run inference
            wav, sr, spect = f5tts.infer(
                ref_file=tmp_path,
                ref_text=ref_text,
                gen_text=gen_text,
                remove_silence=remove_silence,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                speed=speed,
                cfg_strength=cfg_strength,
                file_wave=None,
                file_spect=None,
                seed=seed,
                set_max_chars=max_chars,
                use_ipa=use_ipa,
            )

            # Write generated wave to bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format="WAV")
            buffer.seek(0)

            return StreamingResponse(buffer, media_type="audio/wav")

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


    # To run the app: uvicorn src.f5_tts.api:app --reload
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
