import os
import sys
import subprocess
import venv
import platform

VENV_NAME = "venv_tesi"

def run_command(cmd, env=None):
    print(f"Esecuzione: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)

def has_nvidia_gpu():
    try:
        # Tenta di eseguire nvidia-smi
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def has_amd_gpu():
    if platform.system() != "Windows":
        return False
    try:
        # Cerca schede video AMD tramite WMI
        output = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"], 
            text=True, 
            stderr=subprocess.STDOUT
        )
        return "AMD" in output.upper() or "RADEON" in output.upper()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_venv():
    print(f"Creazione del virtual environment '{VENV_NAME}' con Python 3.12...")
    if platform.system() == "Windows":
        try:
            # Tenta di usare il launcher di Windows per forzare Python 3.12
            subprocess.check_call(["py", "-3.12", "-m", "venv", VENV_NAME])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Avviso: Impossibile trovare 'py -3.12'. Tentativo con l'eseguibile corrente...")
            venv.create(VENV_NAME, with_pip=True)
    else:
        try:
            subprocess.check_call(["python3.12", "-m", "venv", VENV_NAME])
        except (subprocess.CalledProcessError, FileNotFoundError):
            venv.create(VENV_NAME, with_pip=True)
    print("Virtual environment creato con successo.")

def get_venv_python():
    if platform.system() == "Windows":
        return os.path.join(VENV_NAME, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_NAME, "bin", "python")

def install_dependencies():
    python_exe = get_venv_python()
    
    # Aggiorna pip
    run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    
    print("Ricerca dell'hardware video in corso...")
    is_nvidia = has_nvidia_gpu()
    is_amd = has_amd_gpu()
    
    if is_nvidia:
        print("Rilevata GPU NVIDIA (es. RTX 4090). Installazione di PyTorch con supporto CUDA...")
        run_command([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
    elif is_amd:
        print("Rilevata GPU AMD. Installazione di PyTorch per CPU + torch-directml...")
        run_command([python_exe, "-m", "pip", "install", "torch-directml", "torchvision", "torchaudio"])
    else:
        print("Nessuna GPU NVIDIA o AMD rilevata o piattaforma non supportata. Installazione PyTorch standard (CPU)...")
        run_command([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    
    # Installazione dipendenze di base e per DINOv3
    print("Installazione dipendenze extra per DINO...")
    deps = [
        "requests",
        "Pillow",
        "tqdm",
        "numpy",
        "matplotlib",
        "transformers",
        "huggingface_hub",
        "scikit-learn",
        "accelerate"
    ]
    run_command([python_exe, "-m", "pip", "install"] + deps)
    print("\nInstallazione completata con successo!")

def main():
    if sys.version_info < (3, 12):
        print("Attenzione: Si consiglia di eseguire questo script con Python 3.12.")
        print(f"Versione corrente: {sys.version}")
        # Non blocchiamo, ma avvisiamo
        
    if not os.path.exists(VENV_NAME):
        create_venv()
    else:
        print(f"Virtual environment '{VENV_NAME}' già esistente. Salto la creazione.")
        
    install_dependencies()
    print(f"\nPer attivare l'ambiente, esegui:")
    if platform.system() == "Windows":
        print(f"    .\\{VENV_NAME}\\Scripts\\activate")
    else:
        print(f"    source {VENV_NAME}/bin/activate")

if __name__ == "__main__":
    main()
