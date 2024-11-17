from src.preprocessing import preprocess_handshake
from tqdm import tqdm

def preprocess_data(handshakes):
    processed_handshakes = []
    print("Preprocessing handshakes...")
    with tqdm(total=len(handshakes), desc="Preprocessing") as pbar:
        for handshake in handshakes:
            processed_data = preprocess_handshake(handshake)
            processed_handshakes.append(processed_data)
            pbar.update(1)
    return processed_handshakes