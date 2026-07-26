import os
import torch
import random
from PIL import Image
from torchvision import transforms, datasets
from transformers import AutoImageProcessor, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file

# ==========================================
# CONFIGURAZIONE
# ==========================================
MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
BEST_MODEL_DIR = "./dino_kvasir_model/best_model"
DATASET_DIR = r"..\Datasets\kvasir-dataset-v2"

CLASS_NAMES = [
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
    'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'
]
NUM_CLASSES = len(CLASS_NAMES)
id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")

class DINOv3ForImageClassification(torch.nn.Module):
    def __init__(self, model_id, num_labels, id2label, label2id):
        super().__init__()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        
        self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        hidden_size = self.backbone.config.hidden_sizes[-1]
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        
    def forward(self, pixel_values, labels=None, **kwargs):
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, "attentions", None)
        )

def load_trained_model(device):
    print(f"Inizializzazione DINOv3 e caricamento pesi da {BEST_MODEL_DIR}...")
    processor = AutoImageProcessor.from_pretrained(BEST_MODEL_DIR, trust_remote_code=True)
    
    model = DINOv3ForImageClassification(
        model_id=MODEL_ID,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id
    )
    
    safetensors_path = os.path.join(BEST_MODEL_DIR, "model.safetensors")
    bin_path = os.path.join(BEST_MODEL_DIR, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Non ho trovato pesi in {BEST_MODEL_DIR}")
        
    model.to(device)
    model.eval()
    return model, processor

def get_random_test_image():
    """
    Carica il dataset e riproduce la stessa divisione usata in train_model.py
    per estrarre un'immagine casuale esclusivamente dal Test Set.
    """
    print(f"Lettura del dataset originale da {DATASET_DIR}...")
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    
    # Stesso seed e stesse proporzioni usate in train_model.py (80/10/10)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # test_dataset.indices contiene gli indici globali originali del test set
    random_idx = random.choice(test_dataset.indices)
    image_path, true_label_idx = full_dataset.samples[random_idx]
    true_label_name = CLASS_NAMES[true_label_idx]
    
    return image_path, true_label_name

def predict_image(image_path, true_label_name, model, processor, device):
    print(f"\nInferenza su: {image_path}")
    print(f"Classe Reale (Ground Truth): {true_label_name.upper()}")
    
    image = Image.open(image_path).convert("RGB")
    
    if "shortest_edge" in processor.size and processor.size["shortest_edge"] is not None:
        size = processor.size["shortest_edge"]
    else:
        size = processor.size["height"]
        
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    
    pixel_values = data_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
    predicted_class_id = logits.argmax(-1).item()
    predicted_label = model.id2label[predicted_class_id]
    confidence = probabilities[predicted_class_id].item()
    
    print(f"\n=== Risultato Predizione ===")
    print(f"Classe Predetta: {predicted_label.upper()}")
    print(f"Confidenza (Probabilità): {confidence:.2%}")
    
    if predicted_label == true_label_name:
        print("✅ CORRETTO!")
    else:
        print("❌ SBAGLIATO!")
    
    print("\nTop 3 probabilità:")
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    for i in range(3):
        idx = top3_indices[i].item()
        print(f"- {model.id2label[idx]}: {top3_prob[i].item():.2%}")

def main():
    print("=== DINOv3 Inferenza su Test Set (Kvasir-v2) ===")
    device = get_device()
    
    if not os.path.exists(BEST_MODEL_DIR):
        print(f"\nATTENZIONE: La cartella {BEST_MODEL_DIR} non esiste ancora.")
        return
        
    if not os.path.exists(DATASET_DIR):
        print(f"\nATTENZIONE: Il dataset in {DATASET_DIR} non è stato trovato.")
        return
        
    model, processor = load_trained_model(device)
    
    # Ottieni un'immagine randomicamente SOLO dal set di test
    test_image_path, true_label = get_random_test_image()
    
    predict_image(test_image_path, true_label, model, processor, device)

if __name__ == "__main__":
    main()
