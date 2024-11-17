import os
from scapy.all import rdpcap, Dot11, Dot11Beacon, Dot11Elt, Dot11ProbeReq, Dot11AssoReq, Dot11Auth, Dot11ReassoReq
import numpy as np

def preprocess_handshake(handshake):  # Changed to accept handshake packets directly
    """
    Preprocesses a single synthetic handshake.

    Args:
        handshake: A list of Scapy packets representing the handshake.

    Returns:
        A processed handshake in a format suitable for the AI model (e.g., a NumPy array).
    """
    try:
        # Initialize lists to store extracted features
        ssid = []
        bssid = []
        signal_strength = []
        supported_rates = []
        encryption_type = []

        # Iterate through the packets and extract features
        for packet in handshake:  # Iterate through the provided handshake packets
            if packet.haslayer(Dot11Beacon) or packet.haslayer(Dot11ProbeReq):
                # Extract SSID
                try:
                    ssid.append(packet[Dot11Elt].info.decode())
                except:
                    ssid.append("Unknown")

                bssid.append(packet[Dot11].addr2)
                signal_strength.append(packet[Dot11].dBm_AntSignal)

                # Extract supported rates
                if packet.haslayer(Dot11Elt):
                    if packet[Dot11Elt].ID == 1:  # Supported Rates tag
                        supported_rates.append(packet[Dot11Elt].rates)

                # Extract encryption type (more sophisticated logic)
                if packet.haslayer(Dot11Elt):
                    elt_id = packet[Dot11Elt].ID
                    if elt_id == 48:  # RSN information element
                        rsn_info = packet[Dot11Elt].info
                        if rsn_info and rsn_info[0] == 1:  # WPA2
                            encryption_type.append("WPA2-PSK")
                        elif rsn_info and rsn_info[0] == 2:  # WPA3
                            encryption_type.append("WPA3")
                        else:
                            encryption_type.append("Unknown")
                    else:
                        encryption_type.append("Unknown")

            elif packet.haslayer(Dot11AssoReq) or packet.haslayer(Dot11Auth) or packet.haslayer(Dot11ReassoReq):
                # Extract SSID (similar to above)
                try:
                    ssid.append(packet[Dot11Elt].info.decode())
                except:
                    ssid.append("Unknown")

                # For these packets, BSSID is in addr1
                bssid.append(packet[Dot11].addr1)
                signal_strength.append(packet[Dot11].dBm_AntSignal)

                # Extract supported rates (similar to above)
                if packet.haslayer(Dot11Elt):
                    if packet[Dot11Elt].ID == 1:
                        supported_rates.append(packet[Dot11Elt].rates)

                # Extract encryption type (similar to above)
                if packet.haslayer(Dot11Elt):
                    elt_id = packet[Dot11Elt].ID
                    if elt_id == 48:  # RSN information element
                        rsn_info = packet[Dot11Elt].info
                        if rsn_info and rsn_info[0] == 1:  # WPA2
                            encryption_type.append("WPA2-PSK")
                        elif rsn_info and rsn_info[0] == 2:  # WPA3
                            encryption_type.append("WPA3")
                        else:
                            encryption_type.append("Unknown")
                    else:
                        encryption_type.append("Unknown")

        # --- Numerical Conversion and Padding ---

        # Convert features to numerical format
        ssid_numeric = [[ord(char) for char in word] for word in ssid]
        bssid_numeric = [[int(part, 16) for part in bssid_str.split(':')] for bssid_str in bssid]
        signal_strength_numeric = np.array(signal_strength, dtype=np.float32)
        supported_rates_numeric = [[rate for rate in rate_list] for rate_list in supported_rates]

        # One-hot encode encryption type
        encryption_mapping = {"WPA2-PSK": [1, 0, 0], "WPA3": [0, 1, 0], "Unknown": [0, 0, 1]}
        encryption_numeric = [encryption_mapping[enc] for enc in encryption_type]

        # Pad sequences for consistent length
        max_ssid_len = max(len(seq) for seq in ssid_numeric)
        ssid_numeric = [seq + [0] * (max_ssid_len - len(seq)) for seq in ssid_numeric]

        max_rates_len = max(len(seq) for seq in supported_rates_numeric)
        supported_rates_numeric = [seq + [0] * (max_rates_len - len(seq)) for seq in supported_rates_numeric]

        # Create the processed data array
        processed_data = np.array([
            ssid_numeric,
            bssid_numeric,
            signal_strength_numeric,
            supported_rates_numeric,
            encryption_numeric
        ])

        return processed_data

    except Exception as e:
        raise Exception(f"Error preprocessing handshake: {str(e)}")