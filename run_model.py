import torch
import platform
import subprocess

def get_device():
    """
    Rileva automaticamente il device migliore disponibile.
    Se è presente una GPU NVIDIA, usa CUDA.
    Se è presente una GPU AMD (su Windows), usa DirectML se disponibile.
    Altrimenti, usa CPU.
    """
    # 1. Controlla CUDA (NVIDIA)
    if torch.cuda.is_available():
        print(f"Device NVIDIA CUDA rilevato: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    
    # 2. Controlla DirectML (AMD su Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            print(f"Device DirectML rilevato. Ideale per GPU AMD su Windows.")
            return torch_directml.device()
    except ImportError:
        pass
        
    # 3. Fallback alla CPU
    print("Nessuna GPU accelerata rilevata (CUDA/DirectML non disponibili). Uso CPU.")
    return torch.device("cpu")

def load_dinov3_model(device, model_id="facebook/dinov3-small", token=None):
    """
    Carica il modello DINOv3 (Vision Transformer) e il suo Image Processor da Hugging Face.
    Nota: Se il modello è privato/gated, inserisci il tuo Hugging Face token o fai il login 
    tramite `huggingface-cli login`.
    """
    print(f"Caricamento del modello {model_id} da Hugging Face...")
    
    # Importiamo qui per non rallentare l'avvio o causare errori se non installato
    from transformers import AutoImageProcessor, AutoModel
    
    processor = AutoImageProcessor.from_pretrained(model_id, token=token)
    model = AutoModel.from_pretrained(model_id, token=token)
    
    model.to(device)
    model.eval()
    print("Modello e processor caricati con successo sul device!")
    return model, processor

def main():
    print("=== DINOv3 Hardware-Aware Runner ===")
    device = get_device()
    
    # Inserisci qui l'ID esatto del modello a cui hai ottenuto l'accesso
    model_id = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    
    # Se hai già fatto `huggingface-cli login`, token=None va bene. Altrimenti passalo come stringa.
    model, processor = load_dinov3_model(device, model_id=model_id, token=None)
    
    # Esempio di utilizzo con un tensore dummy (simuliamo un'immagine preprocessata)
    print("\nEsecuzione di un'inferenza di prova...")
    # DINO aspetta input processati dal suo AutoImageProcessor (di solito 3 canali, 224x224)
    # Forma tensore: [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        features = model(dummy_input)
        
    print(f"Shape delle features estratte: {features.shape}")
    print("Tutto funziona correttamente e in maniera isolata!")

if __name__ == "__main__":
    main()
