# Self-Supervised ACDC 2017 Cardiac MRI Segmentation

This repository provides a PyTorch implementation for **label-free/self-supervised pseudo-segmentation** of the ACDC 2017 cardiac MRI dataset.

The code assumes ACDC volumes are available in ED/ES format:

```text
Patient001_ED.nii.gz
Patient001_ES.nii.gz
Patient002_ED.nii.gz
Patient002_ES.nii.gz
...
```

This method does **not require manual labels**. It learns image clusters using:
- U-Net style dense prediction
- reconstruction from soft clusters
- augmentation consistency
- entropy regularization
- spatial total variation smoothness
- cluster balance regularization

Important: this is **unsupervised pseudo-segmentation**, not clinically validated LV/RV/MYO segmentation. The generated labels are cluster IDs and may need post-processing or label mapping.

---

## 1. Create Environment

```bash
conda env create -f environment.yml
conda activate acdc_ssl
```

or:

```bash
conda create -n acdc_ssl python=3.10 -y
conda activate acdc_ssl
pip install -r requirements.txt
```

---

## 2. Dataset Structure

Recommended:

```text
ACDC/
├── training/
│   ├── patient001/
│   │   ├── Patient001_ED.nii.gz
│   │   └── Patient001_ES.nii.gz
│   ├── patient002/
│   └── ...
└── testing/
    ├── patient101/
    │   ├── Patient101_ED.nii.gz
    │   └── Patient101_ES.nii.gz
    └── ...
```

The loader searches recursively, so flat folders also work.

---

## 3. Train

```bash
python train.py \
  --data_dir /path/to/ACDC/training \
  --output_dir ./runs/acdc_ssl \
  --epochs 100 \
  --batch_size 8 \
  --num_clusters 4
```

---

## 4. Inference

```bash
python inference.py \
  --data_dir /path/to/ACDC/testing \
  --checkpoint ./runs/acdc_ssl/checkpoints/best.pth \
  --output_dir ./runs/acdc_ssl/test_segmentations \
  --num_clusters 4
```

Outputs are saved as NIfTI files:

```text
Patient101_ED_ssl_seg.nii.gz
Patient101_ES_ssl_seg.nii.gz
```

---

## 5. Visualize Results

```bash
python visualize.py \
  --image /path/to/Patient101_ED.nii.gz \
  --seg ./runs/acdc_ssl/test_segmentations/Patient101_ED_ssl_seg.nii.gz \
  --slice 5 \
  --output ./example_overlay.png
```

---

## Notes

- `num_clusters=4` is usually chosen to roughly represent background, LV cavity, myocardium, and RV cavity.
- Because no labels are used, cluster IDs are arbitrary.
- For stronger results, use this as pseudo-label generation followed by supervised fine-tuning if labels become available.
