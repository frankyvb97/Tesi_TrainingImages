import os
import csv
import torch
import random
from PIL import Image
from torchvision import transforms, datasets
from transformers import AutoImageProcessor, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# CONFIGURAZIONE
# ==========================================
MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
BEST_MODEL_DIR = "./dino_kvasir_model/best_model"
DATASET_DIR = r"..\Datasets\kvasir-dataset-v2"
RESULTS_DIR = "./results"

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
        return SequenceClassifierOutput(logits=logits)

def load_trained_model(device):
    print(f"Inizializzazione DINOv3 e caricamento pesi da {BEST_MODEL_DIR}...")
    processor = AutoImageProcessor.from_pretrained(BEST_MODEL_DIR, trust_remote_code=True)
    model = DINOv3ForImageClassification(MODEL_ID, NUM_CLASSES, id2label, label2id)
    
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

def get_test_dataset_info():
    """Ricrea lo split 80/10/10 e restituisce tutte le immagini e le label del Test Set."""
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    test_images = []
    test_labels = []
    
    # Preleviamo le singole path originarie dal full_dataset
    for idx in test_dataset.indices:
        image_path, true_label_idx = full_dataset.samples[idx]
        test_images.append(image_path)
        test_labels.append(true_label_idx)
        
    return test_images, test_labels

def evaluate_test_set(model, processor, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    test_images, test_labels_idx = get_test_dataset_info()
    print(f"Inizio valutazione sull'intero Test Set ({len(test_images)} immagini)...")
    
    if "shortest_edge" in processor.size and processor.size["shortest_edge"] is not None:
        size = processor.size["shortest_edge"]
    else:
        size = processor.size["height"]
        
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    
    y_true = []
    y_pred = []
    
    csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
    
    # Prepariamo il file CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_class", "predicted_class", "confidence"])
        
        # tqdm mostrerà la barra di caricamento
        for i in tqdm(range(len(test_images)), desc="Inferenza Test Set"):
            image_path = test_images[i]
            true_label_idx = test_labels_idx[i]
            true_label_name = CLASS_NAMES[true_label_idx]
            
            # Carica immagine
            image = Image.open(image_path).convert("RGB")
            pixel_values = data_transforms(image).unsqueeze(0).to(device)
            
            # Passaggio modello
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                
            predicted_class_id = logits.argmax(-1).item()
            predicted_label = CLASS_NAMES[predicted_class_id]
            confidence = probabilities[predicted_class_id].item()
            
            # Registra dati
            y_true.append(true_label_idx)
            y_pred.append(predicted_class_id)
            
            # Scrivi la riga del file CSV
            writer.writerow([image_path, true_label_name, predicted_label, f"{confidence:.4f}"])
            
    print(f"\nPredizioni salvate con successo in {csv_path}")
            
    # Calcolo Metriche
    print("Calcolo metriche in corso...")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # In una classificazione multi-classe calcolata con macro-average, 
    # la Sensitivity (True Positive Rate) è matematicamente equivalente alla Recall.
    sensitivity = recall
    
    # Salvataggio Metriche in txt
    txt_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== RISULTATI VALUTAZIONE TEST SET ===\n")
        f.write(f"Numero totale immagini testate: {len(test_images)}\n\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {precision:.4f} (Macro Average)\n")
        f.write(f"Recall:      {recall:.4f} (Macro Average)\n")
        f.write(f"Sensitivity: {sensitivity:.4f} (Identica alla Recall nel Macro Average)\n")
        f.write(f"F1 Score:    {f1:.4f} (Macro Average)\n")
        
    print(f"Metriche salvate con successo in {txt_path}")
    
    # Generazione Confusion Matrix
    print("Generazione Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    
    # Aumentiamo la dimensione per accomodare gli 8 nomi lunghi delle classi
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.tight_layout()
    
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"Confusion Matrix salvata come immagine in {cm_path}")
    print("\nValutazione completata! Troverai tutto nella cartella 'results'.")

def main():
    print("=== DINOv3 Valutazione Globale su Test Set (Kvasir-v2) ===")
    device = get_device()
    
    if not os.path.exists(BEST_MODEL_DIR):
        print(f"\nATTENZIONE: La cartella {BEST_MODEL_DIR} non esiste ancora.")
        print("Devi prima completare l'addestramento lanciando train_model.py!")
        return
        
    if not os.path.exists(DATASET_DIR):
        print(f"\nATTENZIONE: Il dataset in {DATASET_DIR} non è stato trovato.")
        return
        
    model, processor = load_trained_model(device)
    evaluate_test_set(model, processor, device)

if __name__ == "__main__":
    main()
