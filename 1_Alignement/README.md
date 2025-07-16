# 1. Alignement

Le dossier `1_Alignement` contient les scripts utilisés pour aligner les textes issus des transcriptions OCR avec leur gold. Il suit la structure suivante : 

```
1_Alignement/
├── alignment.py
├── seuil_myers.py
├── stats.py
└── utils
    ├── align_corr.py
    ├── Myers.py
    ├── process_texts.py
    ├── RETAS.py
    └── semantic_search.py
```

## 1.1. alignment.py

Aligne les textes OCR et gold en utilisant l'algorithme choisi entre méthode RETAS, Diff Algorithm de Myers, et Semantic Search de Debaene et enregistre les phrases alignées avec leur CER dans un fichier csv.

Si l'algorithme choisi est RETAS, les textes alignés sont également enregistrés dans deux fichiers txt.

**Usage :**

```sh
python3 alignment.py -a {myers, retas, semantic} -g GOLD -o OCR
```

__options :__

| Option                   | Description                                        |
|--------------------------|----------------------------------------------------|
| `-a ALGO`, `--ocr ALGO`  | Algorithme d'alignement (myers, retas ou semantic) |
| `-g GOLD`, `--gold GOLD` | Chemin vers le fichier gold                        |
| `-o OCR`, `--ocr OCR`    | Chemin vers le fichier OCR                         |

**Exemple :**

```sh
python3 alignment.py -a retas -g ../Corpus/Extraits/gold_test.txt -o ../Corpus/Extraits/ocr_test_tesseract.txt
```



## 1.2. seuil_myers.py

Détermine le seuil à partir duquel on peut raisonnablement arrêter les comparaisons pour une phrase gold donnée en se basant sur des statistiques sur le nombre de matchs par paire de phrases obtenues avec un autre algorithme d'alignement (dans notre cas RETAS).

**Usage :**

```sh
python3 seuil_myers.py -r RESULTS
```

__options :__

| Option                   | Description                                       |
|--------------------------|---------------------------------------------------|
| `-r RESULTS`, `--results RESULTS` | Chemin vers le fichier csv de résultats  |

**Exemple :**

```sh
python3 seuil_myers.py -r results/Extraits/Aligned_texts/RETAS/results_retas_tesseract.csv
```



## 1.3. stats.py

Analyse les résultats obtenus avec l'algorithme d'alignement et le système OCR choisis. Donne le nombre de paires de phrases alignées, le CER moyen entre les paires de phrases, et un graphique de la distribution des CERs.

**Usage :**

```sh
python3 stats.py -a ALGO -r RESULTS
```

__options :__

| Option                               | Description                                                                             |
|--------------------------------------|-----------------------------------------------------------------------------------------|
| `-a ALGO`, `--algo ALGO`             | Algorithme d'alignement utilisé : myers, retas ou semantic                              |
| `-r RESULTS`, `--results RESULTS`    | Chemin vers le fichier csv de résultats obtenus avec l'algorithme d'alignement choisi   |

**Exemple :**

```sh
python3 stats.py -a retas -r results/Extraits/Aligned_texts/RETAS/results_retas_tesseract.csv
```



## 1.4. RETAS.py

Aligne les textes OCR et gold en utilisant la méthode RETAS, segmente les textes alignés en phrases, calcule les CER entre paires de phrases, et enregistre les textes alignés dans 2 fichiers txt et les phrases alignées avec leur CER dans un fichier csv. Ce script est appelé par `alignment.py` mais peut aussi être exécuté seul.

**Usage :**

```sh
python3 RETAS.py -g GOLD -o OCR
```

__options :__

| Option                   | Description                  |
|--------------------------|------------------------------|
| `-g GOLD`, `--gold GOLD` | Chemin vers le fichier gold  |
| `-o OCR`, `--ocr OCR`    | Chemin vers le fichier OCR   |

**Exemple :**

```sh
python3 RETAS.py -g ../../Corpus/Extraits/gold_test.txt -o ../../Corpus/Extraits/ocr_test_tesseract.txt
```



## 1.5. Myers.py

Segmente les textes OCR et gold en phrases et aligne les phrases en utilisant le diff algorithm de Myers, calcule les CER entre paires de phrases, et enregistre les phrases alignées avec leur CER dans un fichier csv. Ce script est appelé par `alignment.py` mais peut aussi être exécuté seul.

**Usage :**

```sh
python3 Myers.py -g GOLD -o OCR
```

__options :__

| Option                   | Description                  |
|--------------------------|------------------------------|
| `-g GOLD`, `--gold GOLD` | Chemin vers le fichier gold  |
| `-o OCR`, `--ocr OCR`    | Chemin vers le fichier OCR   |

**Exemple :**

```sh
python3 Myers.py -g ../../Corpus/Extraits/gold_test.txt -o ../../Corpus/Extraits/ocr_test_tesseract.txt
```



## 1.6. semantic_search.py

Segmente les textes OCR et gold en phrases et aligne les phrases en utilisant la semantic search de Debaene, calcule les CER entre paires de phrases, et enregistre les phrases alignées avec leur CER dans un fichier csv. Ce script est appelé par `alignment.py` mais peut aussi être exécuté seul.

**Usage :**

```sh
python3 semantic_search.py -g GOLD -o OCR
```

__options :__

| Option                   | Description                  |
|--------------------------|------------------------------|
| `-g GOLD`, `--gold GOLD` | Chemin vers le fichier gold  |
| `-o OCR`, `--ocr OCR`    | Chemin vers le fichier OCR   |

**Exemple :**

```sh
python3 semantic_search.py -g ../../Corpus/Extraits/gold_test.txt -o ../../Corpus/Extraits/ocr_test_tesseract.txt
```



## 1.7. process_texts.py

Charge, sauvegarde et effectue les traitements basiques des textes golg/OCR. Ce script est appelé par `RETAS.py`, `Myers.py` et `semantic_search.py`. Il n'est pas exécutable seul.



## 1.8. align_corr.py

Aligne les phrases sous forme de triplets (gold, ocr, correction) en utilisant la méthode RETAS et enregistre les phrases alignées dans un fichier csv.


**Usage :**

```sh
python3 align_corr.py -f .FILE
```

__options :__

| Option                   | Description                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------------|
| `-f FILE`, `--file FILE` | Chemin vers le fichier csv contenant les phrases originales avec leur transcription et leur correction |


**Exemple :**

```sh
python3 align_corr.py -f ../../3_Correction/Tests_existing_models/Thomas_2024/AJ/results/fine_tuned_models/BART/result_bart-base-ocr-10-epochs-tesseract.csv
```
