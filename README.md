# DenoMAE2.0: Improving Denoising Masked Autoencoders by Classifying Local Patches for Automatic Modulation Classification

[Read the paper on arXiv](https://arxiv.org/pdf/2502.18202)

## Model Architecture
DenoMAE 2.0 consists of:
- A Vision Transformer-based encoder that processes partially masked images
- A lightweight decoder that reconstructs the original image from the encoded representation
- A classification head that predicts the locations of unmasked patches

The model employs a unique training strategy where:
1. Random patches of the input image are masked
2. The encoder processes the partially masked input
3. The decoder reconstructs the original image
4. A classifier predicts which patches were kept during masking

This combined approach forces the model to learn robust feature representations that are useful for both denoising and understanding image structure.

## Installation

### Dependencies

A full list of dependencies is provided in the `requirements.txt` file.

### Setting Up the Environment
```bash
# Clone the repository
git clone https://github.com/atik666/denoMAE2/tree/model
# Change into the project directory
cd model

# Option 1: Install directly with pip
pip install -r requirements.txt

# Option 2: Create and use a virtual environment (recommended)
python -m venv denomae_env

# On Windows
denomae_env\Scripts\activate
# On Linux/Mac
source denomae_env/bin/activate

# Install required packages
pip install -r requirements.txt
```

For GPU acceleration (recommended for training):
- Ensure you have compatible NVIDIA drivers installed
- PyTorch will automatically use available CUDA if configured properly

## Usage

### Pretraining
To pretrain the DenoMAE 2.0 model:

```bash
bash ./run_pretrain.sh
```

You can configure the pretraining parameters in the shell file before running.

Key parameters:
- `train_path`: Path to the training data directory
- `test_path`: Path to the test data directory
- `mask_ratio`: Proportion of image patches to mask (default: 0.75)
- `cls_loss_ratio`: Weight of the classification loss (default: 0.1)

### Fine-tuning
To fine-tune the pretrained model for downstream classification tasks:

```bash
bash ./run_finetune.sh
```

Please modify the shell file to configure your fine-tuning parameters.

Key parameters:
- `train_data_path`: Path to labeled training data directory
- `test_data_path`: Path to labeled test data directory
- `num_classes`: Number of classes for classification
- `pretrained_model_path`: Path to the pretrained DenoMAE 2.0 model

## Model Configuration
The DenoMAE 2.0 model can be configured with the following parameters:

- **Encoder**:
  - `patch_size`: Size of image patches (default: 16)
  - `embed_dim`: Embedding dimension (default: 768)
  - `encoder_depth`: Transformer encoder depth (default: 12)
  - `encoder_num_heads`: Number of attention heads (default: 12)

- **Decoder**:
  - `decoder_embed_dim`: Decoder embedding dimension (default: 512)
  - `decoder_depth`: Transformer decoder depth (default: 8)
  - `decoder_num_heads`: Number of decoder attention heads (default: 8)

## Citation
If you use DenoMAE 2.0 in your research, please cite:
```
@misc{faysal2025denomae20improvingdenoisingmasked,
      title={DenoMAE2.0: Improving Denoising Masked Autoencoders by Classifying Local Patches}, 
      author={Atik Faysal and Mohammad Rostami and Taha Boushine and Reihaneh Gh. Roshan and Huaxia Wang and Nikhil Muralidhar},
      year={2025},
      eprint={2502.18202},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.18202}, 
}
```