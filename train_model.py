import os
import sys
import torch
import numpy as np
import evaluate
import shutil
import glob
import json
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel, TrainerCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings

# Disabilita il noioso warning di fallback su CPU di DirectML durante le operazioni lerp dell'AdamW
warnings.filterwarnings("ignore", category=UserWarning, message=".*aten::lerp.Scalar_out.*")

# ==========================================
# CONFIGURAZIONE
# ==========================================
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
    "PATIENCE": 10,
    "AUGMENTATION": {
        "RANDOM_HORIZONTAL_FLIP": True,
        "RANDOM_VERTICAL_FLIP": True,
        "RANDOM_ROTATION": True,
        "COLOR_JITTER": True,
        "RANDOM_RESIZED_CROP": True,
        "ELASTIC_TRANSFORM": True
    }
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
DATASET_DIR = config["DATASET_DIR"]
OUTPUT_DIR = config["OUTPUT_DIR"]

# Parametri di training
NUM_FOLDS = config["NUM_FOLDS"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LEARNING_RATE = config["LEARNING_RATE"]
PATIENCE = config["PATIENCE"]
AUGMENTATION = config.get("AUGMENTATION", {})

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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    print("=== DINOv3 Full Fine-Tuning su Kvasir-v2 ===")
    print("\n[Configurazione Data Augmentation]")
    for key, val in AUGMENTATION.items():
        status = "ATTIVATO" if val else "DISATTIVATO"
        print(f" - {key}: {status}")
    print()
    
    # Salvataggio cartelle fold precedenti nel backup
    if os.path.exists(OUTPUT_DIR):
        backup_dir = os.path.join(OUTPUT_DIR, "backup")
        
        # Pulisci il backup esistente se c'è
        if os.path.exists(backup_dir):
            for item in os.listdir(backup_dir):
                item_path = os.path.join(backup_dir, item)
                for attempt in range(5):
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=False)
                        else:
                            os.remove(item_path)
                        break
                    except Exception as e:
                        if attempt == 4:
                            print(f"  ! Impossibile rimuovere {item_path} dopo 5 tentativi: {e}")
                            print("Interruzione dell'esecuzione per evitare inconsistenze dei dati.")
                            sys.exit(1)
                        else:
                            import time
                            time.sleep(1)
        else:
            os.makedirs(backup_dir, exist_ok=True)
            
        print(f"Salvataggio cartelle precedenti in {backup_dir}...")
        for item in os.listdir(OUTPUT_DIR):
            if item.startswith("fold_"):
                item_path = os.path.join(OUTPUT_DIR, item)
                if os.path.isdir(item_path):
                    backup_path = os.path.join(backup_dir, item)
                    if os.path.exists(backup_path):
                        # Se esiste ancora, fa un ultimo tentativo di cancellazione
                        for attempt in range(5):
                            try:
                                shutil.rmtree(backup_path, ignore_errors=False)
                                break
                            except Exception as e:
                                if attempt == 4:
                                    print(f"  ! Impossibile rimuovere {backup_path} dopo 5 tentativi: {e}")
                                    print("Interruzione dell'esecuzione per evitare inconsistenze dei dati.")
                                    sys.exit(1)
                                else:
                                    import time
                                    time.sleep(1)
                    try:
                        shutil.move(item_path, backup_path)
                        print(f"  - Spostata {item_path} in backup")
                    except Exception as e:
                        print(f"  ! Impossibile spostare {item_path}: {e}")
                        print("Interruzione dell'esecuzione per evitare inconsistenze dei dati.")
                        sys.exit(1)
                    
    device = get_device()
    print(f"Device per il training: {device}")
    
    # 1. Image Processor
    print("\nCaricamento dell'Image Processor...")
    # NOTA: Assicurati di essere autenticato via terminale (huggingface-cli login)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2. Caricamento Dataset (ImageFolder salta i PDF automaticamente)
    print("\nLettura e divisione del dataset...")
    # Creiamo un trasformatore PyTorch usando i parametri del processor HuggingFace
    if "shortest_edge" in processor.size and processor.size["shortest_edge"] is not None:
        size = processor.size["shortest_edge"]
    else:
        size = processor.size["height"]
        
    image_mean = processor.image_mean
    image_std = processor.image_std
    
    # 1. Trasformazioni per il Training (con Data Augmentation configurabile)
    train_transform_list = []
    
    # RandomResizedCrop (Nuovo): Ritaglia un'area casuale dell'immagine e la ridimensiona. Aiuta il modello a ignorare i bordi neri tipici delle endoscopie e a focalizzarsi sui tessuti.
    if AUGMENTATION.get("RANDOM_RESIZED_CROP", False):
        train_transform_list.append(transforms.RandomResizedCrop(size, scale=(0.7, 1.0)))
    else:
        # Fallback al resize standard se il crop random è disattivato
        train_transform_list.append(transforms.Resize((size, size)))
        
    # RandomHorizontalFlip: Inverte l'immagine orizzontalmente con probabilità del 50%. Aumenta la variabilità posizionale.
    if AUGMENTATION.get("RANDOM_HORIZONTAL_FLIP", False):
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
    # RandomVerticalFlip: Inverte l'immagine verticalmente con probabilità del 50%. Aiuta il modello a essere invariante rispetto all'orientamento della sonda.
    if AUGMENTATION.get("RANDOM_VERTICAL_FLIP", False):
        train_transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        
    # RandomRotation: Ruota l'immagine casualmente fino a 30 gradi. Simula le rotazioni imperfette della telecamera endoscopica.
    if AUGMENTATION.get("RANDOM_ROTATION", False):
        train_transform_list.append(transforms.RandomRotation(degrees=30))
        
    # ColorJitter: Altera in modo casuale luminosità, contrasto e saturazione. Estremamente utile per simulare diverse condizioni di illuminazione del viscere.
    if AUGMENTATION.get("COLOR_JITTER", False):
        train_transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        
    # ElasticTransform (Nuovo): Applica deformazioni locali casuali all'immagine. Ideale per simulare la natura elastica e deformabile dei tessuti biologici (es. polipi).
    if AUGMENTATION.get("ELASTIC_TRANSFORM", False):
        train_transform_list.append(transforms.ElasticTransform(alpha=50.0))
        
    # Trasformazioni base sempre necessarie (conversione in Tensore e Normalizzazione)
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
    
    train_transforms = transforms.Compose(train_transform_list)

    # 2. Trasformazioni per Validazione/Test (PULITE, nessuna alterazione)
    val_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    SPLIT_JSON = "config/dataset_split.json"
    dataset_folder_name = os.path.basename(os.path.normpath(DATASET_DIR))
    
    # Costruiamo prima l'ImageFolder temporaneo solo per leggere le classi ed eseguire lo split se non esiste il JSON
    temp_dataset = datasets.ImageFolder(root=DATASET_DIR)
    class_names = temp_dataset.classes
    num_classes = len(class_names)
    print(f"Classi trovate ({num_classes}): {class_names}")
    
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    split_data = None
    if os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            split_data = json.load(f)
            
    if split_data is not None and split_data.get("dataset_folder") == dataset_folder_name:
        print(f"File {SPLIT_JSON} trovato e valido. Caricamento dei file split...")
        train_samples = split_data["train"]
        val_samples = split_data["val"]
        test_samples = split_data["test"]
    else:
        print(f"Nessun file split valido per '{dataset_folder_name}'. Creazione split 80/10/10...")
        train_size = int(0.8 * len(temp_dataset))
        val_size = int(0.1 * len(temp_dataset))
        test_size = len(temp_dataset) - train_size - val_size
        
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            temp_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # Estraiamo i (path, class_idx) e aggiungiamo il nome della classe per maggiore leggibilità
        train_samples = [[temp_dataset.samples[i][0], temp_dataset.samples[i][1], class_names[temp_dataset.samples[i][1]]] for i in train_ds.indices]
        val_samples = [[temp_dataset.samples[i][0], temp_dataset.samples[i][1], class_names[temp_dataset.samples[i][1]]] for i in val_ds.indices]
        test_samples = [[temp_dataset.samples[i][0], temp_dataset.samples[i][1], class_names[temp_dataset.samples[i][1]]] for i in test_ds.indices]
        
        # Salviamo il JSON
        new_split = {
            "dataset_folder": dataset_folder_name,
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }
        with open(SPLIT_JSON, "w", encoding="utf-8") as f:
            json.dump(new_split, f, indent=4)
        print(f"Nuovo split salvato correttamente in {SPLIT_JSON}.")
        
    print(f"Dataset suddiviso: {len(train_samples)} Train, {len(val_samples)} Validation, {len(test_samples)} Test.")

    # Convertiamo i subset in un formato compatibile con l'HF Trainer
    class JSONSubsetDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            sample = self.samples[idx]
            path = sample[0]
            label = sample[1]
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return {"pixel_values": image, "labels": label}
            
    hf_train_dataset = JSONSubsetDataset(train_samples, transform=train_transforms)
    hf_val_dataset = JSONSubsetDataset(val_samples, transform=val_transforms)

    # 3. Caricamento Modello e Configurazione Linear Probing
    print(f"\nCaricamento dell'architettura DINO per classificazione a {num_classes} classi...")
    class DINOv3ForImageClassification(torch.nn.Module):
        def __init__(self, model_id, num_classes, id2label, label2id):
            super().__init__()
            self.num_classes = num_classes
            self.id2label = id2label
            self.label2id = label2id
            
            # Load the backbone
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.config = self.backbone.config
            
            # (Modifica) Sblocco del backbone per effettuare il Full Fine-Tuning
            for param in self.backbone.parameters():
                param.requires_grad = True
                
            # Create classification head
            # Aggiungiamo un Linear layer finale basato sull'hidden size del modello
            hidden_size = self.backbone.config.hidden_sizes[-1]
            self.classifier = torch.nn.Linear(hidden_size, num_classes)
            
        def forward(self, pixel_values, labels=None, **kwargs):
            outputs = self.backbone(pixel_values=pixel_values)
            pooled_output = outputs.pooler_output
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
                
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

    class SaveBestAndLastModelCallback(TrainerCallback):
        def __init__(self, best_model_path, last_model_path, processor, patience):
            self.best_model_path = best_model_path
            self.last_model_path = last_model_path
            self.processor = processor
            self.patience = patience
            self.best_metric = -float('inf')
            self.patience_counter = 0
            self.trainer = None # Verrà assegnato subito dopo l'inizializzazione del Trainer

        def _save_full_checkpoint(self, output_dir, model):
            os.makedirs(output_dir, exist_ok=True)
            # 1. Pesi del modello
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            # 2. Configurazione (indispensabile per alcuni script di caricamento)
            if hasattr(model, "backbone") and hasattr(model.backbone, "config"):
                model.backbone.config.save_pretrained(output_dir)
            # 3. Image Processor
            self.processor.save_pretrained(output_dir)
            
            # Se il trainer è collegato, salviamo tutti gli stati avanzati
            if self.trainer is not None:
                torch.save(self.trainer.args, os.path.join(output_dir, "training_args.bin"))
                self.trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                if self.trainer.optimizer is not None:
                    torch.save(self.trainer.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                if self.trainer.lr_scheduler is not None:
                    torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and "eval_accuracy" in metrics:
                current_accuracy = metrics["eval_accuracy"]
                model = kwargs.get("model")
                
                # 1. Salva SEMPRE il modello corrente come 'last_model' ad ogni fine epoca
                print(f"\n[Callback] Epoca completata! Salvataggio last_model in corso...")
                self._save_full_checkpoint(self.last_model_path, model)
                
                # 2. Se è anche il migliore finora, salvalo come 'best_model' e resetta la patience
                if current_accuracy > self.best_metric:
                    self.best_metric = current_accuracy
                    self.patience_counter = 0
                    print(f"⭐⭐⭐ [NUOVO BEST MODEL] ⭐⭐⭐")
                    print(f"L'accuratezza è migliorata a: {current_accuracy:.4f}!")
                    print(f"Salvataggio in corso in {self.best_model_path}...")
                    self._save_full_checkpoint(self.best_model_path, model)
                else:
                    # 3. Peggioramento: incrementa la patience
                    self.patience_counter += 1
                    print(f"[Callback] Nessun miglioramento (Il migliore resta: {self.best_metric:.4f}).")
                    print(f"Patience: {self.patience_counter}/{self.patience}")
                    if self.patience_counter >= self.patience:
                        print(f"\n[Callback] 🛑 Raggiunto il limite di Patience ({self.patience}). Early Stopping attivato!")
                        control.should_training_stop = True

    # 3. Creazione loop K-Fold
    print(f"\nPreparazione Stratified K-Fold con {NUM_FOLDS} folds...")
    train_val_samples = train_samples + val_samples
    train_val_labels = [sample[1] for sample in train_val_samples]
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_samples, train_val_labels), 1):
        # RESET FORZATO dello stato di Accelerate per consentire multipli avvii di Trainer in loop
        from accelerate.state import AcceleratorState, PartialState
        AcceleratorState._reset_state()
        PartialState._reset_state()
        
        print(f"\n{'='*50}")
        print(f"=== FOLD {fold}/{NUM_FOLDS} ===")
        print(f"{'='*50}")
        
        fold_train_samples = [train_val_samples[i] for i in train_idx]
        fold_val_samples = [train_val_samples[i] for i in val_idx]
        
        hf_train_dataset = JSONSubsetDataset(fold_train_samples, transform=train_transforms)
        hf_val_dataset = JSONSubsetDataset(fold_val_samples, transform=val_transforms)

        # 4. Inizializzazione Processor e Modello da ZERO per evitare Data Leakage
        print("Inizializzazione Processor e Modello...")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        model = DINOv3ForImageClassification(
            MODEL_ID, 
            num_classes=len(class_names), 
            id2label=id2label, 
            label2id=label2id
        )
        model.to(device)

        # 5. Setup Trainer con Callback personalizzato
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        best_model_dir = os.path.join(fold_output_dir, "best_model")
        last_model_dir = os.path.join(fold_output_dir, "last_model")
        
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="no", 
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            push_to_hub=False,
            dataloader_pin_memory=False,
        )

        # HACK: Il Trainer di HuggingFace forza il fallback su CPU se non trova CUDA o MPS.
        TrainingArguments.device = property(lambda self: device)
        
        # Invece di fare monkey-patching globale su Accelerator (che corrompe l'inizializzazione 
        # dello stato nei Fold successivi), creiamo esplicitamente l'Accelerator e forziamo
        # il device nell'istanza.
        from accelerate import Accelerator
        from accelerate.state import PartialState
        accel = Accelerator()
        accel.state._shared_state["device"] = device
        PartialState._shared_state["device"] = device
        
        # HACK VITALE: Diciamo ad HuggingFace Trainer di NON distruggere e ricreare 
        # lo stato appena configurato. Altrimenti in _setup_devices cancellerà _shared_state.
        training_args.accelerator_config.use_configured_state = True
        # Assegniamo comunque lo stato (anche se ignorato in alcune versioni, è good practice)
        training_args.distributed_state = accel.state

        custom_callback = SaveBestAndLastModelCallback(best_model_dir, last_model_dir, processor, PATIENCE)
        
        class DirectMLTrainer(Trainer):
            def _prepare_inputs(self, inputs):
                # Forza in modo assoluto e manuale lo spostamento di ogni tensore sulla GPU (DML)
                prepared = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        prepared[k] = v.to(device)
                    else:
                        prepared[k] = v
                return prepared

        trainer = DirectMLTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_val_dataset,
            processing_class=processor,
            compute_metrics=compute_metrics,
            callbacks=[custom_callback]
        )
        custom_callback.trainer = trainer

        # 6. Esecuzione Addestramento Fold
        print(f"\nInizio addestramento per Fold {fold}!")
        trainer.train()
        
        # 7. Salvataggio last_model finale Fold
        print(f"\nAddestramento Fold {fold} completato! Salvataggio last_model in: {last_model_dir}")
        custom_callback._save_full_checkpoint(last_model_dir, model)
        
        # Pulizia della memoria
        del model
        del trainer
        del custom_callback
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\nProcesso Stratified K-Fold completato con successo per tutti i fold!")

if __name__ == "__main__":
    main()
