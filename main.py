import os
from src.data_generator import generate_synthetic_data
from src.data_processor import preprocess_data
from src.trainer import train_models
import string

# --- Data Generation Parameters ---
num_handshakes = 10000
ssid_prefix = "SyntheticWiFi_"
bssid_prefix = "00:11:22:33:"

# --- Create Character Space ---
char_space = string.ascii_letters + string.digits + string.punctuation

# --- Generate Synthetic Data ---
handshakes, passwords = generate_synthetic_data(num_handshakes, ssid_prefix, bssid_prefix, char_space)

# --- Preprocess the Handshakes ---
processed_handshakes = preprocess_data(handshakes)

# --- Train the Models ---
train_models(processed_handshakes, passwords, char_space)