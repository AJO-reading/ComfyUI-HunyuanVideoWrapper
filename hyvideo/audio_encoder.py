import torch
import torch.nn as nn
import whisper
import librosa
import numpy as np
from typing import Optional, Union, Tuple
import logging
import os
try:
    import comfy.model_management as mm
except ImportError:
    print("ComfyUI model management not available")

log = logging.getLogger(__name__)

class WhisperAudioEncoder:
    def __init__(self, model_name="tiny", device=None):
        if device is None and 'mm' in globals():
            device = mm.get_torch_device()
        elif device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_name = model_name
        
        try:
            self.model = whisper.load_model(model_name, device=device)
            self.model.eval()
            log.info(f"Loaded Whisper {model_name} model on {device}")
        except Exception as e:
            log.error(f"Failed to load Whisper model: {e}")
            raise
            
    def extract_features(self, audio_input):
        if isinstance(audio_input, str):
            audio = whisper.load_audio(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.cpu().numpy()
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        with torch.no_grad():
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            features = self.model.encoder(mel)
            
        return features

class AudioNet(nn.Module):
    def __init__(self, audio_dim=512, hidden_dim=3072, num_heads=24):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, audio_features, video_features, audio_strength=0.8):
        audio_proj = self.audio_proj(audio_features)
        attn_output, _ = self.cross_attention(
            query=video_features,
            key=audio_proj,
            value=audio_proj
        )
        
        aligned_features = self.layer_norm(video_features + attn_output)
        aligned_features = self.output_proj(aligned_features)
        aligned_features = video_features + audio_strength * (aligned_features - video_features)
        
        return aligned_features

def create_audio_conditioning(audio_features, audio_strength=0.8, device=None):
    if device is None and 'mm' in globals():
        device = mm.get_torch_device()
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return {
        "audio_features": audio_features.to(device),
        "audio_strength": torch.tensor(audio_strength, device=device, dtype=torch.float32),
        "audio_condition": True,
        "has_audio": True
    }
