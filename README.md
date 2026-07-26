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

