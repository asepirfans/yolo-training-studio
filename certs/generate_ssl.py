import os
import shutil
import socket
import subprocess
import sys

CERTS_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_PATH = os.path.join(CERTS_DIR, "cert.pem")
KEY_PATH = os.path.join(CERTS_DIR, "key.pem")
CNF_PATH = os.path.join(CERTS_DIR, "openssl.cnf")

def find_openssl():
    from_path = shutil.which("openssl")
    if from_path:
        return from_path
    common_locations = [
        r"C:\Program Files\Git\usr\bin\openssl.exe",
        r"C:\Program Files (x86)\Git\usr\bin\openssl.exe",
        r"C:\OpenSSL-Win64\bin\openssl.exe",
        r"C:\OpenSSL-Win32\bin\openssl.exe",
    ]
    for loc in common_locations:
        if os.path.exists(loc):
            return loc
    return "openssl"

def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

def generate_self_signed_cert():
    lan_ip = get_lan_ip()
    print(f"[SSL Generator] Membuat sertifikat SSL untuk YOLO Training Studio...")
    
    cnf_content = f"""[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = ID
ST = Jakarta
L = Jakarta
O = YOLO Training Studio
OU = Vision Studio
CN = {lan_ip}

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = {lan_ip}
IP.2 = 127.0.0.1
"""
    with open(CNF_PATH, "w", encoding="utf-8") as f:
        f.write(cnf_content)

    openssl_bin = find_openssl()

    cmd = [
        openssl_bin,
        "req", "-x509", "-nodes",
        "-days", "3650",
        "-newkey", "rsa:2048",
        "-keyout", KEY_PATH,
        "-out", CERT_PATH,
        "-config", CNF_PATH
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[SSL Generator] Sukses!")
        print(f"  - Certificate: {CERT_PATH}")
        print(f"  - Private Key: {KEY_PATH}")
        print(f"  - SAN: IP {lan_ip}, 127.0.0.1, localhost")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[SSL Generator] Error generating certificate: {e.stderr}")
        return False
    except Exception as e:
        print(f"[SSL Generator] Exception: {e}")
        return False

if __name__ == '__main__':
    generate_self_signed_cert()
