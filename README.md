# DM-RSC
[TCSVT]: Multi-Frame Rolling Shutter Correction with Diffusion Models

# README

## 1. Data Preparation

To train and test the models, please first download the datasets:

- Download **Fastec** and **Carla** datasets from the repositories:
  - [DeepUnrollNet](https://github.com/ethliup/DeepUnrollNet)
- and download the **BSRSC** dataset from the same repository above.
  - [BSRSC](https://github.com/ljzycmd/BSRSC)


---


## 3. How to Test

### Download Pretrained Checkpoints

Download the pretrained model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1LfSP8Z8s8Ofv56LaDXeCVeDNdy3quP6v?usp=sharing).

### Sampling from MDM

Run the following command to generate samples from the MDM model:

```bash
python -m script.sample_mdm --config configs/MDM/fastec_sample-MDM.yaml

```

### Sampling from ODM

Run the following command to generate samples from the ODM model:

```bash
python -m script.sample_odm --config configs/ODM/fastec_sample_ODM.yaml

```

## 3. How to Train

### Train MDM Model

Run the following command to train the MDM model:

```bash
python -m script.train_mdm --config DM-RSC/configs/MDM/fastec_train_MDM.yaml
```

### Train ODM Model

Run the following command to train the ODM model:

```bash
python -m script.train_odm --config DM-RSC/configs/ODM/fastec_train_ODM.yaml
```

## 4. Supplementary Materials

Download supplementary materials from [Google Drive](https://drive.google.com/drive/folders/1LfSP8Z8s8Ofv56LaDXeCVeDNdy3quP6v?usp=sharing).

Refer to the `supplementary.zip` archive for comparison GIF and additional results.

