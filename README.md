# istego100k++
# IStego100K++: A Large-Scale GAN-Based Steganalysis Dataset

> Official implementation and dataset generation code for the paper:
> **"IStego100K++: A Large-Scale GAN-Based Steganalysis Dataset
>  Using Adversarially Learned Embedding Costs"**
> Saif Salah Al-Din Affat — University of Anbar, Iraq

---

## Overview
IStego100K++ is generated using a pre-trained UT-GAN generator
applied to 100,071 cover images from IStego100K at 1024×1024
resolution, with random payload rates (0.1–0.5 bpp) and JPEG
quality factors (75–95).

**Results:**
- Mean PSNR: 74.52 dB
- Mean SSIM: 0.9994
- DCTR/GFR detection accuracy: ≈ 50% (undetectable)

---

## Requirements
pip install -r requirements.txt

---

## Usage
python UTGAN_complete.py

---

## Citation
If you use this code, please cite:
[Paper citation here after publication]
