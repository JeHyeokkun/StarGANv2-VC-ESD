#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import glob
import yaml
import numpy as np
import torch
import librosa
import soundfile as sf
import torchaudio

from munch import Munch
from parallel_wavegan.utils import load_model

from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder


# -----------------------
# Audio / mel utilities
# -----------------------
def build_mel_transform(n_mels=80, n_fft=2048, win_length=1200, hop_length=300):
    return torchaudio.transforms.MelSpectrogram(
        n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )

def preprocess_wave_to_mel(wave, to_mel, mean=-4.0, std=4.0):
    wave_t = torch.from_numpy(wave).float()
    mel = to_mel(wave_t)  # [n_mels, T]
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std  # [1, n_mels, T]
    return mel


# -----------------------
# Model builders
# -----------------------
def build_starganv2(model_params):
    args = Munch(model_params)
    generator = Generator(
        args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel
    )
    mapping_network = MappingNetwork(
        args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim
    )
    style_encoder = StyleEncoder(
        args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim
    )
    nets = Munch(generator=generator, mapping_network=mapping_network, style_encoder=style_encoder)
    return nets


# -----------------------
# Style helpers
# -----------------------
def compute_style_from_pairs(style_pairs, starganv2, device, sr, to_mel):
    """
    style_pairs: dict[key] = (ref_path_or_empty, domain_int)
      - if path == "", style is sampled from mapping_network latent
      - else path -> style_encoder(ref_mel, label)
    returns: dict[key] = (style_tensor, label_tensor)
    """
    ref_embeds = {}
    for key, (p, dom) in style_pairs.items():
        label = torch.LongTensor([dom]).to(device)
        if not p:
            latent_dim = starganv2.mapping_network.shared[0].in_features
            with torch.no_grad():
                style = starganv2.mapping_network(torch.randn(1, latent_dim).to(device), label)
        else:
            wave, wav_sr = librosa.load(p, sr=None, mono=True)
            if wav_sr != sr:
                wave = librosa.resample(wave, orig_sr=wav_sr, target_sr=sr)
            wave, _ = librosa.effects.trim(wave, top_db=30)
            mel = preprocess_wave_to_mel(wave, to_mel).to(device)
            with torch.no_grad():
                style = starganv2.style_encoder(mel.unsqueeze(1), label)  # [1, style_dim]
        ref_embeds[key] = (style, label)
    return ref_embeds


# -----------------------
# ESD speaker mapping
# -----------------------
def esd_spk_to_domain(spk_id_str):
    # 0011..0020 -> 0..9
    spk = str(spk_id_str).zfill(4)
    if not (spk.isdigit() and 11 <= int(spk) <= 20):
        raise ValueError("target_speaker must be 0011..0020")
    return int(spk) - 11

def any_wav_from_esd_root(esd_root, spk):
    paths = glob.glob(os.path.join(esd_root, spk, "**", "*.wav"), recursive=True)
    paths = [p for p in paths if os.path.isfile(p)]
    random.shuffle(paths)
    return paths[0] if paths else None


# -----------------------
# Main inference
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="StarGANv2-VC ESD inference (24kHz)")
    parser.add_argument("--config", type=str, required=True, help="Path to StarGANv2-VC config.yml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--source", type=str, required=True, help="Path to source wav")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--ref", type=str, default="", help="Path to reference wav (style-encoder mode). If empty, use mapping mode.")
    parser.add_argument("--target_speaker", type=str, default="0015", help="ESD speaker id 0011..0020")
    parser.add_argument("--esd_root", type=str, default="", help="Optional ESD root to auto-pick a reference file when --ref is empty")
    parser.add_argument("--vocoder", type=str, default="Vocoder/checkpoint-400000steps.pkl", help="Parallel WaveGAN vocoder checkpoint")
    parser.add_argument("--sr", type=int, default=24000, help="Sampling rate for inference")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save_name", type=str, default="converted.wav", help="Output wav filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Build mel transform from config (must match training)
    sp = cfg.get("preprocess_params", {}).get("spect_params", {})
    to_mel = build_mel_transform(
        n_mels=80 if sp.get("n_mels") is None else sp.get("n_mels"),
        n_fft=sp.get("n_fft", 2048),
        win_length=sp.get("win_length", 1200),
        hop_length=sp.get("hop_length", 300),
    )

    # Build F0 model
    f0_model = JDCNet(num_class=1, seq_len=192)
    f0_params = torch.load("Utils/JDC/bst.t7")["net"]
    f0_model.load_state_dict(f0_params)
    f0_model = f0_model.eval().to(device)

    # Build vocoder
    vocoder = load_model(args.vocoder).to(device).eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()

    # Build StarGANv2-VC
    nets = build_starganv2(cfg["model_params"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    weights = ckpt.get("model_ema", ckpt.get("model", ckpt))
    for k in nets:
        nets[k].load_state_dict(weights[k])
        nets[k] = nets[k].eval().to(device)

    # Load source wav
    src_wave, src_sr = librosa.load(args.source, sr=None, mono=True)
    if src_sr != args.sr:
        src_wave = librosa.resample(src_wave, orig_sr=src_sr, target_sr=args.sr)
    src_wave = src_wave.astype(np.float32)
    if np.max(np.abs(src_wave)) > 0:
        src_wave = src_wave / np.max(np.abs(src_wave))

    source_mel = preprocess_wave_to_mel(src_wave, to_mel).to(device)

    # Decide target domain and style
    dom = esd_spk_to_domain(args.target_speaker)

    # If ref not given and esd_root provided, auto-pick a ref file from that speaker
    ref_path = args.ref
    if not ref_path and args.esd_root:
        candidate = any_wav_from_esd_root(args.esd_root, args.target_speaker)
        if candidate:
            ref_path = candidate

    # Prepare style
    style_pairs = {"target": (ref_path, dom)}  # "" -> mapping mode, non-empty -> style encoder mode
    ref_embeds = compute_style_from_pairs(style_pairs, nets, device, args.sr, to_mel)
    style, label = ref_embeds["target"]

    # Forward
    with torch.no_grad():
        f0_feat = f0_model.get_feature_GAN(source_mel.unsqueeze(1))
        out_mel = nets.generator(source_mel.unsqueeze(1), style, F0=f0_feat)  # [1, 80, T]
        c = out_mel.transpose(-1, -2).squeeze().to(device)                    # [T, 80]
        y_out = vocoder.inference(c).view(-1).cpu().numpy()

    # Save
    out_path = os.path.join(args.output, args.save_name)
    sf.write(out_path, y_out, args.sr, subtype="PCM_16")
    print(f"[OK] Saved: {out_path}")

    # Optional: also save trimmed source and ref copies for audit
    try:
        sf.write(os.path.join(args.output, "_src_24k.wav"), src_wave, args.sr, subtype="PCM_16")
        if ref_path:
            ref_wav, ref_sr = librosa.load(ref_path, sr=None, mono=True)
            if ref_sr != args.sr:
                ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=args.sr)
            sf.write(os.path.join(args.output, "_ref_24k.wav"), ref_wav, args.sr, subtype="PCM_16")
    except Exception as e:
        print(f"[WARN] Audit save failed: {e}")


if __name__ == "__main__":
    main()
