import os
from funcLabelSampleGeneration import generate_constellation_images
from tqdm import tqdm
from typing import List, Tuple

def generate_constellations(
    samples_per_image: int,
    image_size: Tuple[int, int],
    SNR_dB: List[float],  # Changed to List[float]
    image_num: List[int],
    mod_types: List[str],
    mode: str,
    base_path: str
) -> None:
    """
    Generate labeled constellation images for various modulation types and SNR values.
    
    Args:
    samples_per_image (int): Number of samples to produce each constellation image
    image_size (Tuple[int, int]): Size of the output images
    SNR_dB (List[float]): List of SNR values in dB
    image_num (List[int]): Number of images to generate per modulation type
    mod_types (List[str]): List of modulation types
    mode (str): 'train' or 'test'
    base_path (str): Base directory to store generated data
    """
    # Iterate through each SNR value
    for snr in tqdm(SNR_dB, desc="Processing SNR values"):
        fold_path = os.path.join(base_path, 'labeled', f'{snr}_dB', mode)
        os.makedirs(fold_path, exist_ok=True)

        # Generate images for each modulation type
        for mod in tqdm(mod_types, desc=f"Generating images for SNR {snr}dB"):
            generate_constellation_images(
                mod, samples_per_image, image_num[0], image_size, snr, fold_path
            )
    
    print("Processing complete for all SNR values.")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'samples_per_image': 1024, # number of data points per image
        'image_size': (224, 224),
        'image_num': [100], # total number of images to generate: image_num * len(mod_types)
        'mod_types': ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK'],
        'mode': 'train',
        'SNR_dB': [-1, -2, -3],  # List of SNR values
        'base_path': '/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data'
    }

    generate_constellations(**CONFIG)
