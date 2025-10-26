#!/usr/bin/env python3
"""
StarGANv2-VC Voice Conversion Script
Converts input audio to match reference speaker's style using style encoder only.
"""

import argparse
import os
import sys
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import time

# Add parent directory to path to find Utils and models modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder

# Source: http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is18/en_speaker_used.txt
# Source: https://github.com/jjery2243542/voice_conversion
speakers = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def load_wav_safe(path, sr=24000):
    """Safely load audio file with error handling"""
    try:
        y, in_sr = sf.read(path, dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if in_sr != sr:
            y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
        return y, sr
    except Exception:
        y, _sr = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), sr

def compute_style_from_reference(reference_path, speaker_id,starganv2):
    """Compute style embedding from reference audio"""
    wave, sr = load_wav_safe(reference_path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    mel_tensor = preprocess(wave).to('cuda')
    with torch.no_grad():
        label = torch.LongTensor([speaker_id]).to('cuda')
        ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
    return ref, label

def load_models():
    """Load all required models"""
    print("Loading F0 model...")
    F0_model = JDCNet(num_class=1, seq_len=192)
    f0_model_path = os.path.join(parent_dir, "Utils/JDC/bst.t7")
    params = torch.load(f0_model_path)['net']
    F0_model.load_state_dict(params)
    F0_model.eval()
    F0_model = F0_model.to('cuda')
    
    print("Loading vocoder...")
    from parallel_wavegan.utils import load_model
    vocoder_path = os.path.join(parent_dir, "Vocoder/checkpoint-400000steps.pkl")
    vocoder = load_model(vocoder_path).to('cuda').eval()
    vocoder.remove_weight_norm()
    
    print("Loading StarGANv2 model...")
    model_path = os.path.join(parent_dir, 'Models/ESD_PVC/epoch_00150.pth')
    config_path = os.path.join(parent_dir, 'Models/ESD_PVC/config_PVC.yml')
    with open(config_path) as f:
        starganv2_config = yaml.safe_load(f)
    
    starganv2 = build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')
    
    return F0_model, vocoder, starganv2

def convert_voice(input_path, reference_path, output_path):
    """Convert input voice to match reference speaker style"""
    
    # Load models
    F0_model, vocoder, starganv2 = load_models()
    
    # Load input audio
    print(f"Loading input audio: {input_path}")
    audio, source_sr = librosa.load(input_path, sr=24000)
    audio = audio / np.max(np.abs(audio))
    audio = audio.astype(np.float32)
    
    # Load reference audio and compute style
    print(f"Loading reference audio: {reference_path}")
    # For simplicity, assume speaker ID 16 (can be made configurable)
    speaker_id = speakers.index(16) if 16 in speakers else 0
    ref, label = compute_style_from_reference(reference_path, speaker_id, starganv2)
    
    # Convert audio
    print("Converting voice...")
    start = time.time()
    
    source = preprocess(audio).to('cuda:0')
    
    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
        out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)
        
        c = out.transpose(-1, -2).squeeze().to('cuda')
        y_out = vocoder.inference(c)
        y_out = y_out.view(-1).cpu().numpy()
    
    end = time.time()
    print(f'Total processing time: {end - start:.3f} sec')
    
    # Save output
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    sf.write(output_path, y_out, 24000)
    print(f"Converted audio saved to: {output_path}")
    
    return y_out

def main():
    parser = argparse.ArgumentParser(description='StarGANv2-VC Voice Conversion')
    parser.add_argument('input_wav', help='Path to input audio file')
    parser.add_argument('reference_wav', help='Path to reference audio file')
    parser.add_argument('--output', '-o', default='./converted_output.wav', 
                       help='Output file path (default: ./converted_output.wav)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.input_wav):
        print(f"Error: Input file '{args.input_wav}' not found")
        return 1
    
    if not os.path.exists(args.reference_wav):
        print(f"Error: Reference file '{args.reference_wav}' not found")
        return 1
    
    try:
        # Convert voice
        convert_voice(args.input_wav, args.reference_wav, args.output)
        print("Voice conversion completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
