import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
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

# Parametri di training (Linear Probing)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-4

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

    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=data_transforms)
    
    # Mappatura classi
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classi trovate ({num_classes}): {class_names}")
    
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    # Split 80/20 train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset caricato: {train_size} immagini di train, {val_size} immagini di validazione.")

    # Convertiamo i subset in un formato compatibile con l'HF Trainer
    # Il Trainer accetta liste di dict o oggetti dataset compatibili.
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            image, label = self.subset[idx]
            return {"pixel_values": image, "labels": label}
            
    hf_train_dataset = CustomDataset(train_dataset)
    hf_val_dataset = CustomDataset(val_dataset)

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
    
    # 4. Setup Trainer
    print("\nInizializzazione Training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        eval_strategy="epoch", # Nuova sintassi rispetto a evaluation_strategy
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        compute_metrics=compute_metrics,
    )

    # 5. Esecuzione
    print("\nInizio del ciclo di addestramento! (Visualizzerai il progresso nella barra qui sotto)")
    trainer.train()
    
    # 6. Salvataggio modello finale
    print(f"\nAddestramento completato! Salvataggio del modello migliore in: {OUTPUT_DIR}/best_model")
    trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    print("Modello e processor esportati correttamente.")

if __name__ == "__main__":
    main()
