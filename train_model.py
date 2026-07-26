import os
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

# ==========================================
# CONFIGURAZIONE
# ==========================================
MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
# Assicurati che il path sia corretto rispetto a dove lanci lo script
DATASET_DIR = r"..\Datasets\kvasir-dataset-v2"
OUTPUT_DIR = "./dino_kvasir_model"

# Parametri di training (Linear Probing ottimizzato per 4GB VRAM)
BATCH_SIZE = 16          # Ridotto da 32 a 16 per evitare out-of-memory (OOM) su 4GB VRAM
EPOCHS = 50              # Aumentato, l'addestramento verrà fermato dall'Early Stopping
LEARNING_RATE = 1e-3     # Aumentato (standard per un linear probing più reattivo)
PATIENCE = 5             # Epoche di tolleranza senza miglioramento prima dello stop

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
    print("=== DINOv3 Linear Probing su Kvasir-v2 ===")
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
    
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    SPLIT_JSON = "dataset_split.json"
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
            
    hf_train_dataset = JSONSubsetDataset(train_samples, transform=data_transforms)
    hf_val_dataset = JSONSubsetDataset(val_samples, transform=data_transforms)

    # 3. Caricamento Modello e Configurazione Linear Probing
    print(f"\nCaricamento dell'architettura DINO per classificazione a {num_classes} classi...")
    class DINOv3ForImageClassification(torch.nn.Module):
        def __init__(self, model_id, num_labels, id2label, label2id):
            super().__init__()
            self.num_labels = num_labels
            self.id2label = id2label
            self.label2id = label2id
            
            # Load the backbone
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.config = self.backbone.config
            
            # Freeze backbone for linear probing
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Create classification head
            hidden_size = self.config.hidden_sizes[-1]
            self.classifier = torch.nn.Linear(hidden_size, num_labels)
            
        def forward(self, pixel_values, labels=None, **kwargs):
            outputs = self.backbone(pixel_values=pixel_values, **kwargs)
            pooled_output = outputs.pooler_output
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=getattr(outputs, "attentions", None)
            )

    model = DINOv3ForImageClassification(
        model_id=MODEL_ID,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    # 4. Setup Trainer con Callback personalizzato (Evita totalmente di creare checkpoint-* su disco)
    print("\nInizializzazione Training...")
    
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    last_model_dir = os.path.join(OUTPUT_DIR, "last_model")
    
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
            
            # Se il trainer è collegato, salviamo tutti gli stati avanzati (Optimizer, Scheduler, ecc.)
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
                print(f"\n[Callback] Epoca completata! Aggiornamento in {self.last_model_path}...")
                self._save_full_checkpoint(self.last_model_path, model)
                
                # 2. Se è anche il migliore finora, salvalo come 'best_model' e resetta la patience
                if current_accuracy > self.best_metric:
                    self.best_metric = current_accuracy
                    self.patience_counter = 0
                    print(f"[Callback] Nuovo best model trovato (Accuracy: {current_accuracy:.4f})! Salvataggio in {self.best_model_path}...")
                    self._save_full_checkpoint(self.best_model_path, model)
                else:
                    # 3. Peggioramento: incrementa la patience
                    self.patience_counter += 1
                    print(f"[Callback] Nessun miglioramento (Migliore: {self.best_metric:.4f}). Patience: {self.patience_counter}/{self.patience}")
                    if self.patience_counter >= self.patience:
                        print(f"\n[Callback] 🛑 Raggiunto il limite di Patience ({self.patience}). Early Stopping attivato!")
                        control.should_training_stop = True
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="no", # Blocca la creazione delle cartelle checkpoint di HuggingFace!
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        push_to_hub=False,
        dataloader_pin_memory=False,
    )

    custom_callback = SaveBestAndLastModelCallback(best_model_dir, last_model_dir, processor, PATIENCE)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[custom_callback]
    )
    # Agganciamo il trainer al callback in modo che possa accedere all'optimizer e scheduler
    custom_callback.trainer = trainer

    # 5. Esecuzione
    print("\nInizio del ciclo di addestramento! (Visualizzerai il progresso nella barra qui sotto)")
    trainer.train()
    
    # 6. Salvataggio last_model finale
    print(f"\nAddestramento completato! Salvataggio dell'ultimo modello elaborato in: {last_model_dir}")
    custom_callback._save_full_checkpoint(last_model_dir, model)

if __name__ == "__main__":
    main()
