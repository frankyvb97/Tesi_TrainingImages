import sys
import subprocess
import platform

def run_command(cmd):
    print(f"Esecuzione: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def has_nvidia_gpu():
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"], 
                text=True, 
                stderr=subprocess.STDOUT
            )
            if "NVIDIA" in output.upper():
                return True
        except Exception:
            pass
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def has_amd_gpu():
    if platform.system() != "Windows":
        return False
    try:
        output = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"], 
            text=True, 
            stderr=subprocess.STDOUT
        )
        return "AMD" in output.upper() or "RADEON" in output.upper()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("=== Utility di Configurazione GPU per DINOv3 ===")
    
    # Verifica di essere nell'ambiente virtuale
    if sys.prefix == sys.base_prefix:
        print("[AVVISO] Sembra che tu non sia nell'ambiente virtuale (venv_tesi).")
        print("Ti consigliamo di attivarlo con '.\\venv_tesi\\Scripts\\activate' prima di lanciare questo script.")
        print("Premi INVIO per continuare a tuo rischio, oppure CTRL+C per annullare.")
        try:
            input()
        except KeyboardInterrupt:
            sys.exit(1)

    print("\nRicerca dell'hardware video in corso...")
    is_nvidia = has_nvidia_gpu()
    is_amd = has_amd_gpu()

    python_exe = sys.executable

    if is_nvidia:
        print("\n[Rilevata GPU NVIDIA]")
        print("1. Pulizia pacchetti in conflitto (torch-directml)...")
        subprocess.call([python_exe, "-m", "pip", "uninstall", "-y", "torch-directml"])
        
        print("2. Installazione di PyTorch CUDA (per GPU NVIDIA)...")
        run_command([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
        print("\n[OK] Switch completato! Ora il tuo ambiente è configurato per NVIDIA.")
        
    elif is_amd:
        print("\n[Rilevata GPU AMD / Intel]")
        print("1. Installazione di PyTorch base per supporto DirectML...")
        run_command([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        
        print("2. Installazione del bridge torch-directml...")
        run_command([python_exe, "-m", "pip", "install", "torch-directml"])
        print("\n[OK] Switch completato! Ora il tuo ambiente è configurato per AMD/Intel via DirectML.")
        
    else:
        print("\n[Nessuna GPU accelerata rilevata (o sistema non supportato)]")
        print("Installazione di PyTorch standard (CPU mode)...")
        subprocess.call([python_exe, "-m", "pip", "uninstall", "-y", "torch-directml"])
        run_command([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        print("\n[OK] Switch completato! Ora il tuo ambiente è configurato per l'esecuzione su CPU.")
        
    print("\nInstallazione dipendenze extra dal file requirements.txt (se presenti)...")
    import os
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        run_command([python_exe, "-m", "pip", "install", "-r", req_path])

if __name__ == "__main__":
    main()
