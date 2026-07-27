# Progetto di Tesi: Classificazione di Immagini Mediche (Kvasir-v2) con DINOv3

Questo repository contiene il codice e l'ambiente per il progetto di tesi relativo all'addestramento e alla valutazione di un modello di classificazione di immagini mediche (dataset Kvasir-v2). Il cuore del progetto si basa sull'utilizzo del foundation model **DINOv3 (ConvNeXt-Tiny)** di Meta AI, specializzato mediante la tecnica del **Linear Probing**.

## Struttura della Cartella

- **`train_model.py`**: Lo script principale del progetto. Si occupa di caricare il dataset, inizializzare il modello DINOv3 con i pesi pre-addestrati scaricati da Hugging Face, congelare il backbone (per il Linear Probing) e addestrare un nuovo layer di classificazione finale a 8 classi (corrispondenti a Kvasir-v2).
- **`run_model.py`**: Script (presumibilmente) adibito all'esecuzione dell'inferenza o alla valutazione del modello sui dati di test una volta completato l'addestramento.
- **`setup_env.py`**: Script di utility per la configurazione e il setup iniziale dell'ambiente.
- **`requirements.txt`**: File contenente l'elenco esatto delle dipendenze Python e le loro versioni necessarie per far girare correttamente il codice senza conflitti.
- **`venv_tesi/`**: La directory contenente l'ambiente virtuale Python in cui sono installati i pacchetti del progetto.
- **`dino_kvasir_model/`**: La cartella generata automaticamente dallo script di addestramento. Contiene i checkpoint salvati dal `Trainer` di Hugging Face e il modello esportato con i pesi ottimali.

---

## Istruzioni di Avvio e Setup

Il progetto include degli script intelligenti per configurare automaticamente l'ambiente in base all'hardware rilevato, garantendo la totale assenza di conflitti tra pacchetti (es. CUDA vs DirectML). Puoi scegliere se usare il metodo **Automatico** (consigliato) o quello **Manuale**.

### Metodo 1: Setup Automatico (Consigliato)
Questo script creerà automaticamente l'ambiente virtuale (`venv_tesi`), rileverà la tua scheda video (NVIDIA o AMD) e installerà i pacchetti corretti in totale autonomia.

1. Apri il terminale nella root del progetto (`Progetto_Tesi`).
2. Esegui il setup automatico:
   ```bash
   python setup_env.py
   ```
3. Attiva l'ambiente virtuale appena creato:
   ```bash
   .\venv_tesi\Scripts\activate
   ```

### Gestione Cambio Hardware (`switch_GPU.py`)
Se l'ambiente è già installato, ma trasferisci il progetto su un altro PC (o cambi scheda video da NVIDIA ad AMD o viceversa), puoi usare l'utility inclusa per adattare l'ambiente senza dover reinstallare tutto da zero:
1. Attiva l'ambiente (`.\venv_tesi\Scripts\activate`).
2. Esegui lo script:
   ```bash
   python switch_GPU.py
   ```
Lo script disinstallerà in modo sicuro i pacchetti in conflitto e re-installerà la libreria PyTorch corretta per il tuo nuovo hardware.

---

### Metodo 2: Setup Manuale
Se preferisci avere controllo granulare sull'installazione:

1. Crea l'ambiente virtuale ed attivalo:
   ```bash
   python -m venv venv_tesi
   .\venv_tesi\Scripts\activate
   ```
2. Installa la libreria PyTorch adatta al tuo sistema (es. CUDA per NVIDIA):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   *(Nota: Se hai una scheda AMD, puoi omettere `--index-url` e installare `torch-directml`).*
3. Installa i restanti pacchetti base:
   ```bash
   pip install -r requirements.txt
   ```

---

### Avvio dell'Addestramento (Train)
Per avviare l'addestramento e generare il modello Kvasir-DINOv3, eseguire:
```bash
python train_model.py
```

### Avvio dell'Inferenza (Run Model)
Per avviare la valutazione del Test Set e generare le metriche e la matrice di confusione nella cartella `/results`:
```bash
python run_model.py
```

---

## Log delle Modifiche e Correzioni (Changelog)

Questa sezione documenta tutti i fix architetturali, di dipendenze e di codice affrontati e risolti durante lo sviluppo. *Questo registro verrà aggiornato costantemente ad ogni nuova correzione.*

### [Luglio 2026] Risoluzione Conflitti di Dipendenze e Fix Architettura DINOv3

1. **Risoluzione Errore `DTensor` e Conflitti PyTorch/Transformers:**
   - **Problema:** Lo script andava in crash tentando di caricare `DTensor` durante l'inizializzazione dei moduli distribuiti in Hugging Face.
   - **Causa:** Conflitto tra versioni vecchie di PyTorch (2.3.1 e 2.4.1) e il codice interno di `transformers==5.14.1` (o librerie notturne/aggiornate) che richiede implicitamente un backend compatibile con PyTorch >= 2.5.0.
   - **Soluzione:** È stato effettuato l'upgrade globale all'ambiente aggiornando `torch>=2.5.0`, `torchvision>=0.20.0` e `torchaudio>=2.5.0`.

2. **Risoluzione Lock sui file `torch` su Windows (`WinError 32`):**
   - **Problema:** L'aggiornamento/installazione di pip falliva a causa del blocco file dell'ambiente (spesso dovuto al Language Server di IDE come VSCode/Pylance).
   - **Soluzione:** Abbiamo rinominato temporaneamente la vecchia cartella della libreria `torch` bloccata per permettere al resolver di `pip` di completare una nuova installazione pulita del pacchetto 2.5 senza conflitti.

3. **Incompatibilità SciPy / NumPy v2 (`numpy has no attribute 'long'`):**
   - **Problema:** PyTorch richiede moduli compilati per `numpy<2.0.0` per evitare crash di compatibilità, ma `scipy` nella sua vecchia versione si scontrava con `numpy`.
   - **Soluzione:** È stato forzato il downgrade di `numpy` a `1.26.4` (ossia `<2.0.0`) e contemporaneamente `scipy` è stato aggiornato (`>=1.14.1`) per essere perfettamente compatibile con la versione `1.x` di `numpy`, stabilizzando il Dataloader.

4. **Implementazione di una Classe Wrapper per `AutoModelForImageClassification`:**
   - **Problema:** L'esecuzione crashava con l'errore `Unrecognized configuration class [...] per AutoModelForImageClassification`.
   - **Causa:** Il repository remoto di DINOv3 (`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`) su Hugging Face non possiede un'implementazione nativa di classificazione per immagini per il flag `trust_remote_code=True` in Hugging Face (dispone solo del backbone di feature extraction base).
   - **Soluzione:** Il file `train_model.py` è stato refattorizzato rimuovendo `AutoModelForImageClassification` in favore di un wrapper custom (`DINOv3ForImageClassification`). La classe carica il backbone base, lo congela per il Linear Probing, e istanzia in modo esplicito un `torch.nn.Linear(hidden_size, num_labels)` per calcolare logit e `CrossEntropyLoss`.

5. **Fix dell'Estrazione Dimensioni Immagini (`shortest_edge`):**
   - **Problema:** `processor.size["shortest_edge"]` lanciava un `TypeError` (restituendo `NoneType`), causando un errore nel blocco `transforms.Resize`.
   - **Soluzione:** Corretto il codice per effettuare il fallback su `processor.size["height"]` nel caso in cui `shortest_edge` non sia definito nel dizionario restituito da `AutoImageProcessor`.

Tutti i requisiti sono stati cristallizzati in `requirements.txt` per replicare perfettamente l'ambiente funzionante. Lo script ora raggiunge correttamente le iterazioni del ciclo di addestramento su CPU/acceleratore.

6. **Configurazione di Git e `.gitignore`:**
   - **Richiesta:** Inizializzazione repository Git, esclusione di cartelle pesanti e primo commit/push.
   - **Azione:** Creato il file `.gitignore` per escludere il virtual environment (`venv_tesi/`) e i salvataggi dei modelli (`dino_kvasir_model/`) dal tracciamento. Git locale è pronto per il commit iniziale ("Prima versione del progetto"). *Nota: gli step di push automatici non verranno eseguiti dal bot senza esplicita richiesta manuale futura.*

7. **Risoluzione Warning `pin_memory` del Dataloader:**
   - **Problema:** Durante il training veniva sollevato l'avviso `UserWarning: 'pin_memory' argument is set as true but no accelerator is found`.
   - **Causa:** Il `Trainer` di Hugging Face imposta di default `dataloader_pin_memory=True` (utile per accelerare il passaggio dei dati verso la VRAM della GPU). Quando il dispositivo in uso è la CPU, PyTorch restituisce un warning perché l'operazione non è supportata o necessaria.
   - **Soluzione:** Aggiunto il parametro `dataloader_pin_memory=False` all'interno dell'oggetto `TrainingArguments` in `train_model.py`.

8. **Creazione Test Set (Split 80/10/10):**
   - **Richiesta:** Isolare un set di immagini "Test" puro, mai visto dal modello né in addestramento né in validazione.
   - **Soluzione:** Modificato lo script di addestramento per suddividere casualmente il dataset con proporzioni 80% (Train), 10% (Val), 10% (Test), impostando un generatore pseudo-casuale con "seed fisso" (es. `42`). In questo modo lo script di inferenza `run_model.py` può ricreare la stessa divisione matematica ed estrarre le immagini di test senza il rischio di pescare immagini già studiate dal modello in fase di validazione.

9. **Gestione Checkpoint (Salvataggio selettivo):**
   - **Richiesta:** Evitare di riempire il disco con decine di cartelle `checkpoint-X` ad ogni epoca e tenere solamente il miglior modello e l'ultimo calcolato.
   - **Soluzione:** È stato impostato `save_total_limit=1` nei `TrainingArguments` in modo che il `Trainer` mantenga in memoria fisica un unico checkpoint intermedio (oltre al best). Alla fine dello script, una funzione di post-processing pulisce l'intero ambiente: esporta i pesi perfetti in `best_model`, rinomina l'ultimissimo checkpoint del ciclo in `last_model` ed elimina permanentemente tutte le altre cartelle residue con la libreria `shutil`.

10. **Refactoring dello script di Validazione / Inferenza Globale:**
    - **Richiesta:** Valutare in automatico l'intero 10% del dataset di test isolato in precedenza, estrarre le predizioni e generare i report prestazionali completi, salvando il tutto in una cartella esterna (`results`).
    - **Soluzione:** Lo script `run_model.py` è stato completamente riscritto. Ora riproduce la divisione del dataset e, invece di processare una sola immagine random, itera su tutte le immagini del *Test Set*. Estrae un registro CSV (`predictions.csv`) contenente per ogni file: Percorso originale, Classe Reale, Classe Predetta e Confidenza. Sulla base di questo calcola Accuracy, Precision, Recall, Sensitivity (che nel caso del calcolo multi-classe Macro Average coincide matematicamente con la Recall) ed F1 Score, esportandole su un file di testo (`results.txt`). Infine crea un plot visivo con la **Confusion Matrix** generata tramite `matplotlib` e `scikit-learn` (`confusion_matrix.png`). Tutto viene salvato ordinatamente in `/results`.

11. **Persistenza dello Split in JSON (`dataset_split.json`):**
    - **Richiesta:** Salvare la ripartizione matematica (Train, Val, Test) in modo permanente su disco, per evitare che un seed matematico generi subset imprecisi se la cartella originale delle immagini dovesse variare nel tempo.
    - **Soluzione:** `train_model.py` scansiona il dataset; se `dataset_split.json` non esiste o appartiene a un dataset dal nome differente, ricalcola i set (80/10/10) e salva i percorsi file esatti all'interno del JSON. Un costrutto `JSONSubsetDataset` sostituisce la libreria `ImageFolder` standard. Lo script `run_model.py` va ora a leggere esattamente da quel file JSON i percorsi delle immagini di Test da valutare, garantendo una separazione stagna a vita.

12. **Implementazione Stratified K-Fold Cross Validation (5-Fold):**
    - **Richiesta:** Unire i dataset di Train e Validation per applicare una Stratified K-Fold in modo da sfruttare appieno i dati di training e ottenere una validazione più robusta.
    - **Soluzione:** Lo script `train_model.py` è stato completamente refattorizzato per dividere iterativamente `train_val_samples` in 5 Fold bilanciati usando `StratifiedKFold` di *scikit-learn*. Ogni iterazione re-inizializza il modello DINOv3 e il processor da zero (per evitare *data leakage*) e salva il proprio `best_model` e `last_model` in cartelle separate (es. `dino_kvasir_model/fold_1/best_model`).

13. **Integrazione Ensemble Inference nel Testing (Soft Voting):**
    - **Richiesta:** Adottare le metodologie della letteratura scientifica per massimizzare le metriche sul Test Set tramite i 5 modelli generati dalla K-Fold.
    - **Soluzione:** In `run_model.py` è stato implementato l'approccio *Ensemble*. Lo script ora carica automaticamente tutti i modelli salvati nei vari Fold e processa ogni immagine del Test Set attraverso la commissione di reti. I tensori Softmax estratti vengono accumulati e si calcola la media probabilistica matematica. I risultati (CSV, metriche e Confusion Matrix) vengono generati individualmente per ciascun Fold e per il super-modello Ensemble, suddivisi in sottocartelle dentro `/results`.

14. **Bugfix Compatibilità Architettura ConvNeXt:**
    - **Problema:** Errore `AttributeError` durante l'inizializzazione del modello a causa di costrutti incompatibili ereditati dai classici Vision Transformer (`cls_token` e `hidden_size`).
    - **Soluzione:** Ripristinata la corretta mappatura tensoriale specifica per il backbone `ConvNeXt-Tiny` integrato in questa variante di DINOv3, adottando la property `hidden_sizes[-1]` e passando tramite il `pooler_output` del Global Average Pooling, sia nello script di addestramento che in quello di inferenza.

15. **Centralizzazione delle Configurazioni (`config.json`):**
    - **Richiesta:** Estrarre tutti i parametri e gli iperparametri (Epochs, Batch Size, Learning Rate, Path, ecc.) dai sorgenti per inserirli in un file unico centralizzato.
    - **Soluzione:** Creata una cartella `config/` all'interno della root di progetto. Al suo interno è stato generato il file `config.json` contenente tutte le variabili. Il pre-esistente `dataset_split.json` è stato anch'esso spostato in `config/` per logica di organizzazione. Entrambi gli script `train_model.py` e `run_model.py` sono stati adattati per leggere il JSON di configurazione in fase di avvio (eliminando l'hardcoding), e il file `.gitignore` aggiornato per tracciare correttamente la nuova posizione.

16. **Rimozione Script Batch e Setup Manuale (Ripristino Ambiente Pulito):**
    - **Richiesta:** Rimuovere gli script batch automatizzati (`start_training.bat`, `start_inference.bat`) a causa di potenziali conflitti di inizializzazione della GPU, e documentare l'avvio manuale.
    - **Soluzione:** Gli script ausiliari sono stati eliminati per ripristinare il setup stabile originale. L'avvio ora segue il processo manuale standard.

17. **Risoluzione Bug HuggingFace Trainer su GPU AMD (DirectML):**
    - **Problema:** L'addestramento tramite il `Trainer` di HuggingFace veniva eseguito forzatamente sulla CPU, ignorando la GPU AMD (lasciando la VRAM allo 0%). Interrompendo e riavviando, lo script andava in crash con l'errore `unbox expects Dml at::Tensor`. Inoltre, un nuovo parametro `num_items_in_batch` del Trainer faceva crashare il forward pass del modello.
    - **Causa:** La libreria `accelerate` (cuore del `Trainer`) non riconosce nativamente il backend `privateuseone:0` (DirectML), eseguendo quindi il fallback su CPU e spostando forzatamente i tensori nella RAM di sistema. Questo fallback è stato risolto forzando l'override del device all'interno dei componenti critici di `accelerate`.
    - **Soluzione:** È stato implementato un "Monkey Patch" in `train_model.py` per sovrascrivere la proprietà `device` in `TrainingArguments` e `AcceleratorState`, ingannando il framework per costringerlo a usare il device AMD. È stata inoltre creata una sottoclasse personalizzata `DirectMLTrainer` che, sovrascrivendo `_prepare_inputs()`, inietta fisicamente ogni singolo tensore di input nella VRAM, garantendo un utilizzo massiccio e corretto della GPU AMD. Il parametro non supportato `num_items_in_batch` è stato bloccato prima che raggiungesse il backbone.

18. **Risoluzione crash loop `AcceleratorState` con DirectML (Troubleshooting Completo):**
    - **Problema:** L'utilizzo ripetuto di HuggingFace `Trainer` in un ciclo K-Fold su Windows con GPU AMD (DirectML) causava il crash al secondo Fold: `AttributeError: AcceleratorState object has no attribute distributed_type`. In altri casi portava a fallimenti silenziosi in cui i tensori finivano su CPU scatenando l'errore `tensor.device().type() == at::DeviceType::PrivateUse1 INTERNAL ASSERT FAILED`.
    - **Diagnostica e Test:** Attraverso una serie di script PyTorch isolati per simulare il comportamento interno di `accelerate` e `transformers.Trainer`, è emerso che:
      1. Il monkey-patching globale di `TrainingArguments.device` e `Accelerator.device` corrompeva l'inizializzazione del pattern Singleton (Borg) di `accelerate`.
      2. Anche reimpostando manualmente l'istanza di `Accelerator()` tra un fold e l'altro, il `Trainer` chiamava in automatico una proprietà interna (`args._setup_devices`) prima dell'addestramento.
      3. Di default, la proprietà `_setup_devices` esegue `AcceleratorState._reset_state(reset_partial_state=True)`, piallando via (cancellando) ogni configurazione manuale del device che avevamo appena preparato per il DirectML, causando un fallimento a cascata nei moduli `PartialState`.
    - **Soluzione:** 
      - Rimozione del monkey-patching globale.
      - Per ogni Fold, l'oggetto `Accelerator` viene istanziato esplicitamente e il suo dizionario interno (`_shared_state["device"]`), così come quello del `PartialState`, vengono popolati a mano con `privateuseone:0`.
      - **Fix Fondamentale:** È stata iniettata la direttiva `training_args.accelerator_config.use_configured_state = True`. Questo parametro vitale impone ad HuggingFace di NON azzerare lo stato di `accelerate` durante il `_setup_devices`, preservando così il nostro setup DirectML per tutti i successivi step del K-Fold.
      - Aggiunta della classe `DirectMLTrainer` che sovrascrive `_prepare_inputs` per assicurare matematicamente lo spostamento di ogni tensore sulla GPU, evitando crash nelle operazioni di convoluzione (`F.conv2d`).

19. **Fix Caricamento Pesi Modello per Inferenza:** 
    - **Problema:** Lo script `run_model.py` andava in crash lanciando l'errore `UnpicklingError: Weights only load failed` causato da un blocco di sicurezza di PyTorch sul ripristino di un tensore custom (`_rebuild_device_tensor_from_numpy`) quando `weights_only` è impostato a True.
    - **Soluzione:** Essendo file .bin generati localmente dall'utente al termine dell'addestramento e quindi completamente sicuri, il vincolo è stato disabilitato impostando esplicitamente `torch.load(..., weights_only=False)` permettendo un'inizializzazione liscia dell'inferenza in Ensemble.

20. **Implementazione Data Augmentation Avanzata:** 
    - **Problema:** Un modello complesso come DINOv3, allenato su un limitato numero di immagini mediche per molte epoche, soffre nativamente di Overfitting (impara a memoria le immagini invece di astrarne i pattern).
    - **Soluzione:** È stata divisa la pipeline di trasformazione in due. Un flusso `train_transforms` aggressivo applicato solo al training (Flip orizzontali/verticali casuali, Rotazione casuale fino a 30 gradi, e Color Jitter per sfalsare luminosità e contrasto tipici della sonda endoscopica) e un flusso `val_transforms` pulito per una validazione rigorosa senza alterazioni.

21. **Creazione Automatica e Fallback della Configurazione:**
    - **Problema:** Rischio di crash immediato se il progetto viene clonato su un nuovo ambiente senza la directory `config/` o senza il file `config.json`.
    - **Soluzione:** Sia `train_model.py` che `run_model.py` sono stati dotati di una funzione di autodetezione. Se `config.json` non viene trovato, il sistema ricrea istantaneamente la cartella e genera un file `.json` compilato con tutti i parametri di default operativi, rendendo il progetto 100% "plug-and-play".
