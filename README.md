# Mémoire sur la correction post-OCR

Ce dépôt contient les scrips utilisés pour mon mémoire sur la correction post-OCR. Ces scripts sont situés dans le dossier [Code](./Code/) et sont organisés par étape.

```
Code/
├── 1_Alignement
│   ├── alignment.py
│   ├── seuil_myers.py
│   ├── stats.py
│   └── utils
├── 2_Analyse_erreurs
│   ├── analyse_errors.py
│   ├── identify_errors.py
│   └── utils
└── 3_Correction
    ├── Analyses
    ├── Prepare_data
    ├── Tests_existing_models
    └── Tokenisation
```



## **1. Alignement**

Le dossier [1_Alignement](./Code/1_Alignement/) contient les scripts utilisés pour l'alignement des textes issus des transcriptions OCR avec le gold, ainsi que les phrases issues des corrections des modèles avec leurs phrases OCR et gold correscpondantes. Plus d'infos [ici](./Code/1_Alignement/README.md).


### Scripts principaux

**1.1. [alignment.py](./Code/1_Alignement/alignment.py) :** Alignement de textes gold/OCR avec l'algorithme choisi entre méthode RETAS, Diff Algorithm de Myers, et Semantic Search de Debaene (exécutable)

**1.2. [seuil_myers.py](./Code/1_Alignement/seuil_myers.py) :** Détermination d'un seuil pour optimiser l'algorithme de Myers (exécutable)

**1.3. [stats.py](./Code/1_Alignement/stats.py) :** Analyse des résultats d'un alignement (exécutable)


### Scripts utiles

**1.4. [RETAS.py](./Code/1_Alignement/utils/RETAS.py) :** Alignement de textes gold/OCR avec la méthode RETAS (exécutable, appelé par `alignment.py`)

**1.5. [Myers.py](./Code/1_Alignement/utils/Myers.py) :** Alignement de textes gold/OCR avec l'algorithme de Myers (exécutable, appelé par `alignment.py`)

**1.6. [semantic_search.py](./Code/1_Alignement/utils/semantic_search.py) :** Alignement de textes gold/OCR avec la recherche sémantique (exécutable, appelé par `alignment.py`)

**1.7. [process_texts.py](./Code/1_Alignement/utils/process_texts.py) :** Chargement, sauvegarde et traitements basiques des textes golg/OCR (appelé par `RETAS.py`, `Myers.py` et `semantic_search.py`)

**1.8. [align_corr.py](./Code/1_Alignement/utils/align_corr.py) :** Alignement des phrases sous forme de triplets (gold, ocr, correction) en utilisant la méthode RETAS (exécutable)




## **2. Analyse des erreurs d'OCR**

Le dossier [2_Analyse_erreurs](./Code/2_Analyse_erreurs/) contient les scripts utilisés pour analyser les erreurs d'OCR. Plus d'infos [ici](./Code/2_Analyse_erreurs/README.md).


### Scripts principaux

**2.1. [identify_errors.py](./Code/2_Analyse_erreurs/identify_errors.py) :** Identification des différents types d'erreurs introduites par un système OCR : séquences insérées, manquantes ou mal transcrites (exécutable)

**2.2. [analyse_errors.py](./Code/2_Analyse_erreurs/analyse_errors.py) :** Analyse des erreurs précises introduites par un système OCR en filtrant par caractères ou par longueur (exécutable)




## **3. Correction**

Le dossier [3_Correction](./Code/3_Correction/) contient les scripts utilisés pour corriger les erreurs d'OCR. Plus d'infos [ici](./Code/3_Correction/README.md).


### 3.0. Préparation des données

**3.0.1. [prepare_data.py](./Code/3_Correction/Prepare_data/prepare_data.py) :** Préparation des phrases alignées pour y appliquer un modèle de correction (exécutable)

**3.0.2. [train_test_split.py](./Code/3_Correction/Prepare_data/train_test_split.py) :** Séparation des données nettoyées en ensemble d'entraînement et de test (exécutable)



### 3.1. Thomas, 2024

Adaptation de LLMs (fine-tuning de BART + instruction-tuning de Llama-2) sur des articles de journaux britanniques (en anglais) du 19ème siècle

**3.1.1. [corr_bart.py](./Code/3_Correction/Tests_existing_models/Thomas_2024/AJ/corr_bart.py) :** Application d'un modèle de correction fine-tuné à partir de BART (exécutable)

**3.1.2. [corr_llama-2.py](./Code/3_Correction/Tests_existing_models/Thomas_2024/AJ/corr_llama-2.py) :** Application d'un modèle de correction instruction-tuné à partir de Llama-2 (exécutable)

**3.1.3. [train_bart.py](./Code/3_Correction/Tests_existing_models/Thomas_2024/AJ/train_bart.py) :** Entraînement de BART pour de la correction d'erreurs d'OCR (exécutable)

**3.1.4. [train_llama-2.py](./Code/3_Correction/Tests_existing_models/Thomas_2024/AJ/train_llama-2.py) :** Entraînement de Llama-2 pour de la correction d'erreurs d'OCR (exécutable)

**3.1.5. [corr_fine_tuned_bart.py](./Code/3_Correction/Tests_existing_models/Thomas_2024/AJ/corr_fine_tuned_bart.py) :** Application d'un modèle de correction fine-tuné à partir de BART sur une partie du roman Auriol de W. Harrison Ainsworth (exécutable)

**3.1.6. [corr_fine_tuned_llama-2.py](./Code/3_Correction/Tests_existing_models/Thomas_2024/AJ/corr_fine_tuned_llama-2.py) :** Application d'un modèle de correction instruction-tuné à partir de Llama-2 sur une partie du roman Auriol de W. Harrison Ainsworth (exécutable)



### 3.2. Etude de la tokenisation

Etude de l'impact de la tokenisation des modèles sur les corrections qu’ils proposent

**3.2.1. [tokenisation.py](./Code/3_Correction/Tokenisation/tokenisation.py) :** Tokenisation des textes avec nltk et les tokenisers des modèles (exécutable)

**3.2.2. [token_alignment.py](./Code/3_Correction/Tokenisation/token_alignment.py) :** Alignement des tokens gold et OCR (exécutable)



### 3.3. Analyse des résultats

**3.3.1. [stats_corrections.py](./Code/3_Correction/Analyses/stats_corrections.py) :** Analyse des résultats des corrections obtenues avec un modèle de correction d'erreurs d'OCR (exécutable)

**3.3.2. [analyse_corr.py](./Code/3_Correction/Analyses/analyse_corr.py) :** Analyse des corrections d'un modèle (exécutable)