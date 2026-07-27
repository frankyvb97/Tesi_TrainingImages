import os
import csv
import json
import glob
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
import os

CONFIG_PATH = "config/config.json"
DEFAULT_CONFIG = {
    "MODEL_ID": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "DATASET_DIR": "..\\Datasets\\kvasir-dataset-v2",
    "OUTPUT_DIR": "./dino_kvasir_model",
    "RESULTS_DIR": "./results",
    "NUM_FOLDS": 5,
    "BATCH_SIZE": 16,
    "EPOCHS": 10,
    "LEARNING_RATE": 0.0005,
    "PATIENCE": 10
}

os.makedirs("config", exist_ok=True)
if not os.path.exists(CONFIG_PATH):
    print(f"File di configurazione non trovato. Creazione di {CONFIG_PATH} con i valori di default...")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    config = DEFAULT_CONFIG
else:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

MODEL_ID = config["MODEL_ID"]
ENSEMBLE_DIR = config["OUTPUT_DIR"] 
DATASET_DIR = config["DATASET_DIR"]
RESULTS_DIR = config["RESULTS_DIR"]

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

def load_ensemble_models(device):
    print(f"Ricerca modelli nella directory {ENSEMBLE_DIR}...")
    model_dirs = glob.glob(os.path.join(ENSEMBLE_DIR, "fold_*", "best_model"))
    
    if not model_dirs:
        raise FileNotFoundError(f"Nessuna cartella trovata in {ENSEMBLE_DIR}/fold_*/best_model")
        
    print(f"Trovati {len(model_dirs)} modelli Fold! Inizializzazione in corso...")
    
    models = {}
    processor = None
    
    for m_dir in tqdm(model_dirs, desc="Caricamento Modelli Ensemble"):
        # Estraiamo il nome del fold (es: fold_1) dal path
        fold_name = os.path.basename(os.path.dirname(m_dir))
        
        if processor is None:
            processor = AutoImageProcessor.from_pretrained(m_dir, trust_remote_code=True)
            
        model = DINOv3ForImageClassification(MODEL_ID, NUM_CLASSES, id2label, label2id)
        
        safetensors_path = os.path.join(m_dir, "model.safetensors")
        bin_path = os.path.join(m_dir, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
        else:
            print(f"Attenzione: Nessun peso trovato in {m_dir}")
            continue
            
        model.to(device)
        model.eval()
        models[fold_name] = model
        
    return models, processor

def get_test_dataset_info():
    """Legge lo split dal file config/dataset_split.json e restituisce le immagini e le label del Test Set."""
    SPLIT_JSON = "config/dataset_split.json"
    
    if not os.path.exists(SPLIT_JSON):
        raise FileNotFoundError(f"Il file {SPLIT_JSON} non esiste. Esegui prima train_model.py per generarlo.")
        
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        
    test_samples = split_data["test"]
    
    test_images = []
    test_labels = []
    
    for sample in test_samples:
        path = sample[0]
        label = sample[1]
        # sample[2] sarebbe il nome della classe, utile per l'utente nel JSON ma non strettamente necessario per l'inferenza
        
        test_images.append(path)
        test_labels.append(label)
        
    return test_images, test_labels

def save_evaluation_results(y_true, y_pred, confidences, output_dir, test_images):
    """Salva CSV, TXT e Confusion Matrix per un set specifico di predizioni."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_class", "predicted_class", "confidence"])
        for i in range(len(test_images)):
            writer.writerow([
                test_images[i], 
                CLASS_NAMES[y_true[i]], 
                CLASS_NAMES[y_pred[i]], 
                f"{confidences[i]:.4f}"
            ])
            
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    txt_path = os.path.join(output_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== RISULTATI VALUTAZIONE ===\n")
        f.write(f"Numero totale immagini testate: {len(test_images)}\n\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {precision:.4f} (Macro Average)\n")
        f.write(f"Recall:      {recall:.4f} (Macro Average)\n")
        f.write(f"Sensitivity: {recall:.4f} (Identica alla Recall nel Macro Average)\n")
        f.write(f"F1 Score:    {f1:.4f} (Macro Average)\n")
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def evaluate_all(models_dict, processor, device):
    test_images, test_labels_idx = get_test_dataset_info()
    print(f"Inizio valutazione globale (Singoli Fold + Ensemble) sull'intero Test Set ({len(test_images)} immagini)...")
    
    if "shortest_edge" in processor.size and processor.size["shortest_edge"] is not None:
        size = processor.size["shortest_edge"]
    else:
        size = processor.size["height"]
        
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    
    fold_names = list(models_dict.keys())
    
    # Dizionari per salvare le metriche individuali
    y_pred_dict = {name: [] for name in fold_names}
    conf_dict = {name: [] for name in fold_names}
    
    # Array per l'Ensemble
    y_pred_ensemble = []
    conf_ensemble = []
    y_true = []
    
    for i in tqdm(range(len(test_images)), desc="Inferenza Multipla + Ensemble"):
        image_path = test_images[i]
        true_label_idx = test_labels_idx[i]
        y_true.append(true_label_idx)
        
        image = Image.open(image_path).convert("RGB")
        pixel_values = data_transforms(image).unsqueeze(0).to(device)
        
        ensemble_probs = torch.zeros(1, NUM_CLASSES).to(device)
        
        with torch.no_grad():
            for name, model in models_dict.items():
                outputs = model(pixel_values=pixel_values)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Registriamo la predizione del singolo fold
                pred_idx = probs.argmax(-1).item()
                conf = probs[0, pred_idx].item()
                
                y_pred_dict[name].append(pred_idx)
                conf_dict[name].append(conf)
                
                # Aggiungiamo le probabilità all'accumulatore per l'Ensemble
                ensemble_probs += probs
                
        # Calcolo predizione Ensemble (Media delle probabilità)
        ensemble_probs /= len(models_dict)
        pred_idx_ens = ensemble_probs.argmax(-1).item()
        conf_ens = ensemble_probs[0, pred_idx_ens].item()
        
        y_pred_ensemble.append(pred_idx_ens)
        conf_ensemble.append(conf_ens)
        
    # Fase di Salvataggio
    for name in fold_names:
        print(f"\nSalvataggio risultati per '{name}' in corso...")
        output_dir = os.path.join(RESULTS_DIR, name)
        save_evaluation_results(y_true, y_pred_dict[name], conf_dict[name], output_dir, test_images)
        
    print(f"\nSalvataggio risultati per 'ensemble' in corso...")
    output_dir = os.path.join(RESULTS_DIR, "ensemble")
    save_evaluation_results(y_true, y_pred_ensemble, conf_ensemble, output_dir, test_images)
    
    print("\nValutazione globale completata! Troverai tutto diviso in sottocartelle dentro 'results'.")

def main():
    print("=== DINOv3 Valutazione Multimodello (Kvasir-v2) ===")
    device = get_device()
    print(f"Device per l'inferenza: {device}")
    
    if not os.path.exists(DATASET_DIR):
        print(f"\nATTENZIONE: Il dataset in {DATASET_DIR} non è stato trovato.")
        return
        
    models_dict, processor = load_ensemble_models(device)
    evaluate_all(models_dict, processor, device)

if __name__ == "__main__":
    main()
