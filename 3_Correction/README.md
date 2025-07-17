# 3. Correction des erreurs d'OCR

Le dossier `3_Correction` contient les scripts utilisés pour corriger les erreurs d'OCR. Il suit la structure suivante : 

```
3_Correction/
├── Analyses
│   └── stats_corrections.py
├── Prepare_data
│   ├── prepare_data.py
│   ├── train_test_split.py
│   └── utils
├── Tests_existing_models
│   └── Thomas_2024
└── Tokenisation
    └── tokenisation.py
```

## 3.0. Préparation des données

Le dossier [Prapare_data](./Prepare_data/) contient les scripts utilisés pour préparer les données à être utilisées pour appliquer et entraîner des modèles de correction post-OCR.

### 3.0.1. prepare_data.py

Prépare les phrases alignées dans un fichier csv obtenu avec un algorithme d'alignement (dans notre cas RETAS) pour être utilisées pour appliquer un modèle de correction. 

1. Supprime les séquences insérées de 10 caractères ou plus qui correspondent en réalité à des informations présentes dans le livre numérisé mais pas dans le gold et ne constituent donc pas une réelle erreur d’OCR ; 

2. Normalise les formats de chapitre, soit la mise en majuscule des titres de chapitres bien transcrits ; 

3. Tronque les phrases trop longues (plus de 1024 caractères) non compatibles avec certains modèles ; 

4. Supprime les paires de phrases ayant un CER trop important pour qu’il s’agisse seulement d’erreurs d’OCR (supérieur à 0.8) ; 

5. Ajoute une colonne id associée à chaque paire de phrases.


**Usage :**

```sh
python3 prepare_data.py -f FILE
```

__options :__

| Option                       | Description                                                         |
|------------------------------|---------------------------------------------------------------------|
| `-f FILE`, `--file FILE`     | Chemin vers le fichier contenant les paires de phrases alignées     |


**Exemple :**

```sh
python3 prepare_data.py -f ../data/Tesseract/results_retas_tesseract.csv
```



### 3.0.2. train_test_split.py

Sépare les données nettoyées en ensemble d'entraînement et de test et sauvegarde ces deux ensembles dans deux fichiers csv.

**Usage :**

```sh
python3 train_test_split.py -f FILE -s TEST_SIZE
```

__options :__

| Option                                       | Description                                                                |
|----------------------------------------------|----------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`                     | Chemin vers le fichier contenant les paires de phrases alignées nettoyées  |
| `-s TEST_SIZE`, `--test_size TEST_SIZE`      | Taille des données à conserver pour le test                                |


**Exemple :**

```sh
python3 train_test_split.py -f ../data/Tesseract/tesseract_sentences.csv -s 0.2
```



## 3.1. Thomas_2024

Adaptation de LLMs (fine-tuning de BART + instruction-tuning de Llama-2) sur des articles de journaux britanniques (en anglais) du 19ème siècle

Le dossier [Thomas_2024](./Tests_existing_models/Thomas_2024/) contient les scripts utilisés pour appliquer les modèles de correction proposés par Thomas en 2024. Il suit la structure suivante : 

```
Thomas_2024/
├── AJ
│   ├── corr_bart.py
│   ├── corr_llama-2.py
│   ├── corr_fine_tuned_bart.py
│   ├── corr_fine_tuned_llama-2.py
│   ├── train_bart.py
│   └── train_llama-2.py
└── llms_post-ocr_correction
```

### 3.1.1. corr_bart.py

Applique un modèle de correction fine-tuné à partir de BART, un modèle de langue seq2seq transformer entraîné pour diverses tâches de débruitage.

**Usage :**

```sh
python3 corr_bart.py -f FILE -m {bart-base, bart-large}
```

__options :__

| Option                         | Description                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`       | Chemin vers le fichier csv sur lequel appliquer la correction de phrases alignées    |
| `-m MODEL`, `--model MODEL`    | Nom du modèle ('bart-base' ou 'bart-large')                                          |

Le fichier obtenus est un csv contenant les colonnes id, Ground Truth, OCR Text, Model Correction, old_CER, new_CER, CER_reduction.

__Notes :__

- La valeur `bart-base` pour l'option `model` applique le modèle de correction fine-tuné à partir de la version 'base' du modèle BART (140M paramètres).
- La valeur `bart-large` pour l'option `model` applique le modèle de correction fine-tuné à partir de la version 'large' du modèle BART (400M paramètres).

**Exemple :**

```sh
python3 corr_bart.py -f ../../../data/Tesseract/tesseract_sentences.csv -m bart-base
```



### 3.1.2. corr_llama-2.py

Applique un modèle de correction instruction-tuné à partir de Llama 2.

**Usage :**

```sh
python3 corr_llama-2.py -f FILE -m {llama-2-7b, llama-2-13b}
```

__options :__

| Option                         | Description                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`       | Chemin vers le fichier csv sur lequel appliquer la correction de phrases alignées    |
| `-m MODEL`, `--model MODEL`    | Nom du modèle ('llama-2-7b' ou 'llama-2-13b')                                        |

Le fichier obtenus est un csv contenant les colonnes id, Ground Truth, OCR Text, Model Correction, old_CER, new_CER, CER_reduction.

__Notes :__

- La valeur `llama-2-7b` pour l'option `model` applique le modèle de correction fine-tuné à partir de la version avec 7 milliards de paramètres.
- La valeur `llama-2-13b` pour l'option `model` applique le modèle de correction fine-tuné à partir de la version avec 13 milliards de paramètres.

**Exemple :**

```sh
python3 corr_llama-2.py -f ../../../data/Tesseract/tesseract_sentences.csv -m llama-2-7b
```



### 3.1.3. train_bart.py

Entraîne BART pour de la correction d'erreurs d'OCR.

**Usage :**

```sh
python3 train_bart.py -m {bart-base, bart-large} -c CONFIG -d DATA
```

__options :__

| Option                          | Description                                                                          |
|---------------------------------|--------------------------------------------------------------------------------------|
| `-m MODEL`, `--model MODEL`     | Nom du modèle ('bart-base' ou 'bart-large')                                          |
| `-c CONFIG`, `--config CONFIG`  | Chemin vers le fichier de configuration du modèle                                    |
| `-d DATA`, `--data DATA`        | Chemin vers le fichier csv contenant les données d'antraînement                      |

**Exemple :**

```sh
python3 train_bart.py -m bart-base -c AJ_config.yaml -d ../../../data/Tesseract/tesseract_train.csv
```



### 3.1.4. train_llama-2.py

Entraîne Llama-2 pour de la correction d'erreurs d'OCR.

**Usage :**

```sh
python3 train_llama-2.py -m {llama-2-7b, llama-2-13b} -c CONFIG -d DATA
```

__options :__

| Option                          | Description                                                                          |
|---------------------------------|--------------------------------------------------------------------------------------|
| `-m MODEL`, `--model MODEL`     | Nom du modèle ('llama-2-7b' ou 'llama-2-13b')                                        |
| `-c CONFIG`, `--config CONFIG`  | Chemin vers le fichier de configuration du modèle                                    |
| `-d DATA`, `--data DATA`        | Chemin vers le fichier csv contenant les données d'antraînement                      |

**Exemple :**

```sh
python3 train_llama-2.py -m bart-base -c AJ_config.yaml -d ../../../data/Tesseract/tesseract_train.csv
```



### 3.1.5. corr_fine_tuned_bart.py

Applique un modèle de correction fine-tuné à partir de BART sur une partie du roman Auriol de W. Harrison Ainsworth.

**Usage :**

```sh
python3 corr_fine_tuned_bart.py -f FILE -m MODEL
```

__options :__

| Option                         | Description                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`       | Chemin vers le fichier csv sur lequel appliquer la correction de phrases alignées    |
| `-m MODEL`, `--model MODEL`    | Nom du modèle                                                                        |

Le fichier obtenus est un csv contenant les colonnes id, Ground Truth, OCR Text, Model Correction, old_CER, new_CER, CER_reduction.


**Exemple :**

```sh
python3 corr_fine_tuned_bart.py -f ../../../data/Tesseract/tesseract_sentences.csv -m bart-base-ocr-5-epochs-tesseract
```



### 3.1.6. corr_fine_tuned_llama-2.py

Applique un modèle de correction instruction-tuné à partir de Llama 2 sur une partie du roman Auriol de W. Harrison Ainsworth.

**Usage :**

```sh
python3 corr_fine_tuned_llama-2.py -f FILE -m {llama-2-7b, llama-2-13b}
```

__options :__

| Option                         | Description                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`       | Chemin vers le fichier csv sur lequel appliquer la correction de phrases alignées    |
| `-m MODEL`, `--model MODEL`    | Nom du modèle                                                                        |

Le fichier obtenus est un csv contenant les colonnes id, Ground Truth, OCR Text, Model Correction, old_CER, new_CER, CER_reduction.


**Exemple :**

```sh
python3 corr_fine_tuned_llama-2.py -f ../../../data/Tesseract/tesseract_sentences.csv -m llama-2-7b-ocr-5-epochs-tesseract
```



## 3.2. Etude de la tokenisation

Le dossier [Tokenisation](./Tokenisation/) contient les scripts utilisés pour étudier l'impact de la tokenisation des modèles sur les corrections qu'ils proposent.


### 3.2.1. tokenisation.py

Tokenise les textes avec nltk et les tokenisers des modèles.


**Usage :**

```sh
python3 tokenisation.py -f FILE -m MODEL
```

__options :__

| Option                       | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`     | Chemin vers le fichier csv contenant les phrases alignées                                    |
| `-m MODEL`, `--model MODEL`  | Nom du modèle ('bart-base', 'bart-large', 'llama-2-7b' ou 'llama-2-13b')    |


**Exemple :**

```sh
python3 tokenisation.py -m bart-base -f ../data/Tesseract/tesseract_sentences.csv
```



### 3.2.2. token_alignment.py

Aligne les tokens gold et OCR pour chaque paire de phrases.

**Usage :**

```sh
python3 token_alignment.py -f FILE
```

__options :__

| Option                       | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`     | Chemin vers le fichier json contenant les paires de phrases tokenisées      |


**Exemple :**

```sh
python3 token_alignment.py -f results/Tokenised_sentences/tokenised_sentences_bart_tesseract.json
```



## 3.3. Analyse des résultats

Le dossier [Analyses](./Analyses/) contient les scripts utilisés pour analyser des résultats.


### 3.3.1. stats_corrections.py

Analyse les résultats des corrections obtenues avec un modèle de correction d'erreurs d'OCR.


**Usage :**

```sh
python3 stats_corrections.py -f FILE
```

__options :__

| Option                       | Description                                  |
|------------------------------|----------------------------------------------|
| `-f FILE`, `--file FILE`     | Chemin vers le fichier csv de correction     |


**Exemple :**

```sh
python3 stats_corrections.py -f ../Tests_existing_models/Thomas_2024/AJ/results/existing_models/BART/corr_bart_base_tesseract.csv
```



### 3.3.2. analyse_corr.py

Analyse les corrections d'un modèle.


**Usage :**

```sh
python3 analyse_corr.py -f FILE -c CONTEXT
```

__options :__

| Option                       | Description                                                                                                      |
|------------------------------|------------------------------------------------------------------------------------------------------------------|
| `-f FILE`, `--file FILE`     | Chemin vers le fichier csv contenant les phrases originales avec leur transcription et leur correction alignées  |
| `-c CONTEXT`, `--c CONTEXT`  | Longueur des contextes gauches et droits des séquences à analyser                                                |


Longueur des contextes gauches et droits des séquences à analyser


**Exemple :**

```sh
python3 analyse_corr.py -f analyses/aligned_corrections/aligned_bart-base-ocr-10-epochs-tesseract.csv -c 10
```