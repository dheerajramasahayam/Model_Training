from scapy.all import (Dot11, Dot11Beacon, Dot11Elt, RadioTap, RandMAC, 
                       IP, TCP, UDP, RandIP, fuzz, wrpcap, 
                       Dot11AssoReq, Dot11AssoResp, Dot11Auth)
import random
from tqdm import tqdm

def generate_handshake(ssid, bssid, encryption_type="WPA2-PSK"):
    """
    Generates a synthetic Wi-Fi handshake.
    """

    # --- More Realistic Beacon ---
    dot11 = Dot11(type=0, subtype=8, addr1='ff:ff:ff:ff:ff:ff', addr2=bssid, addr3=bssid)
    beacon = Dot11Beacon(cap='ESS+privacy')
    essid = Dot11Elt(ID='SSID', info=ssid)

    # RSN Information Element (more variability)
    rsn_info = b"\x01\x00"  # RSN Version 1

    # Group Cipher Suite (choose randomly)
    group_ciphers = [
        b"\x00\x0f\xac\x02",  # TKIP
        b"\x00\x0f\xac\x04",  # CCMP (AES)
        b"\x00\x0f\xac\x07"   # GCMP-256
    ]
    rsn_info += random.choice(group_ciphers)

    # Pairwise Cipher Suites (choose 1 or 2 randomly)
    pairwise_ciphers = [
        b"\x00\x0f\xac\x04",  # CCMP (AES)
        b"\x00\x0f\xac\x02"   # TKIP
    ]
    num_pairwise_ciphers = random.randint(1, 2)
    rsn_info += (num_pairwise_ciphers).to_bytes(1, byteorder='big')
    for _ in range(num_pairwise_ciphers):
        rsn_info += random.choice(pairwise_ciphers)

    # Authentication Key Management Suite (PSK)
    rsn_info += b"\x01\x00"  # 1 AKM Suite
    rsn_info += b"\x00\x0f\xac\x02"  # Pre-Shared Key

    # RSN Capabilities (add some randomness)
    capabilities = random.randint(0, 65535)
    rsn_info += capabilities.to_bytes(2, byteorder='big')

    rsn = Dot11Elt(ID='RSNinfo', info=rsn_info)

    # --- More Realistic Handshake ---
    client_mac = RandMAC()

    # Add some fuzzing to the packets (random variations)
    asso_req = fuzz(Dot11AssoReq(cap='ESS+privacy', addr1=bssid, addr2=client_mac, addr3=bssid))
    asso_resp = fuzz(Dot11AssoResp(cap='ESS+privacy', addr1=client_mac, addr2=bssid, addr3=bssid))
    auth_req = fuzz(Dot11Auth(algo=0, seqnum=1, status=0, addr1=bssid, addr2=client_mac, addr3=bssid))
    auth_resp = fuzz(Dot11Auth(algo=0, seqnum=2, status=0, addr1=client_mac, addr2=bssid, addr3=bssid))

    # --- Add Data Packets with Random Content ---
    ip_layer = IP(src=RandIP(), dst=RandIP())
    tcp_layer = TCP(dport=random.randint(1, 65535), flags='S')
    udp_layer = UDP(dport=random.randint(1, 65535))
    data_packet1 = fuzz(ip_layer/tcp_layer/"Random TCP Data")
    data_packet2 = fuzz(ip_layer/udp_layer/"Random UDP Data")

    # Assemble the packets
    packets = [
        RadioTap()/dot11/beacon/essid/rsn,
        RadioTap()/dot11/asso_req,
        RadioTap()/dot11/asso_resp,
        RadioTap()/dot11/auth_req,
        RadioTap()/dot11/auth_resp,
        RadioTap()/data_packet1,
        RadioTap()/data_packet2
    ]

    return packets

def generate_synthetic_data(num_handshakes, ssid_prefix, bssid_prefix, char_space):
    handshakes = []
    passwords = []

    print("Generating synthetic data...")
    with tqdm(total=num_handshakes, desc="Generating Handshakes") as pbar:
        for i in range(num_handshakes):
            ssid = ssid_prefix + str(i)
            bssid = bssid_prefix + ':'.join(random.sample('0123456789abcdef', 2))
            encryption_type = random.choice(["WPA2-PSK", "WPA3"])
            
            # Generate a random password
            password = ''.join(random.choice(char_space) for _ in range(random.randint(8, 16)))
            passwords.append(password)

            handshake_packets = generate_handshake(ssid, bssid, encryption_type)
            handshakes.append(handshake_packets)
            pbar.update(1)

    return handshakes, passwords