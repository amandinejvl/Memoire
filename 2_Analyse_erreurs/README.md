# 2. Analyse des erreurs d'OCR

Le dossier `2_Analyse_erreurs` contient les scripts utilisés pour détecter et analyser les erreurs dans les transcriptions OCR. Il suit la structure suivante : 

```
2_Analyse_erreurs/
├── analyse_errors.py
├── identify_errors.py
└── utils
    ├── filters.py
    ├── find_sequences.py
    └── plot.py
```

## 2.1. identify_errors.py

Identifie les différents types d'erreurs introduites par un système OCR (insertions, suppressions, mauvaises transcriptions).

**Usage :**

```sh
python3 identify_errors.py -g GOLD -o OCR -e {inserted_sequences, missing_sequences, wrong_sequences, proportion}
```

__options :__

| Option                                     | Description                                                                                    |
|--------------------------------------------|------------------------------------------------------------------------------------------------|
| `-g GOLD`, `--gold GOLD`                   | Chemin vers le fichier gold                                                                    |
| `-o OCR`, `--ocr OCR`                      | Chemin vers le fichier OCR                                                                     |
| `-e ERROR_TYPE`, `--error_type ERROR_TYPE` | Type d'erreur à analyser : inserted_sequences, missing_sequences, wrong_sequences, proportion  |


__Notes :__

- La valeur `proportion` pour l'option `error-type` permet d'afficher la distribution des trois types d'erreurs (séquences insérées, les séquences manquantes, et les séquences mal transcrites).
- La valeur `inserted_sequences` pour l'option `error-type` permet d'afficher, dans l'ordre, le nombre de séquences insérées, le nombre de caractères insérés, la distribution des longueurs de séquences insérées, la distribution des caractères uniques insérés, et la distribution des types de caractères uniques insérés.
- La valeur `missing_sequences` pour l'option `error-type` permet d'afficher, dans l'ordre, le nombre de séquences non-transcrites, le nombre de caractères non-transcrits, la distribution des longueurs de séquences non-transcrites, la distribution des caractères uniques non-transcrits, et la distribution des types de caractères uniques non-transcrits.
- La valeur `wrong_sequences` pour l'option `error-type` permet d'afficher, dans l'ordre, le nombre de séquences mal transcrites, le nombre de caractères mal transcrits, la distribution des longueurs de séquences mal transcrites, une heatmap des erreurs de transcription OCR, une heatmap des erreurs de transcription OCR les plus fréquentes, la distribution des caractères uniques mal reconnus, la distribution des caractères uniques transcrits à tort, la distribution des types de caractères uniques mal transcrits, la distribution des types de caractères uniques transcrits à tort.



**Exemple :**

```sh
python3 identify_errors.py -g ../1_Alignement/results/Textes_complets/Aligned_texts/RETAS/Tesseract/retas_gold_aligned_with_ocr_tesseract.txt -o ../1_Alignement/results/Textes_complets/Aligned_texts/RETAS/Tesseract/retas_ocr_tesseract_aligned_with_gold.txt -e proportion
```



## 2.2. analyse_errors.py

Analyse des erreurs précises introduites par un système OCR et renvoie 4 fichiers csv : 

- **sequences_with_gold_context.csv** : séquences insérées / non-transcrites / mal transcrites avec leurs contextes gauches et droits du fichier gold ;
- **sequences_with_ocr_context.csv** : séquences insérées / non-transcrites / mal transcrites avec leurs contextes gauches et droits du fichier OCR ;
- **filtered_sequences_with_gold_context.csv** : séquences insérées / non-transcrites / mal transcrites filtrées selon le nombre de caractères ou un caractère particulier, avec leurs contextes gauches et droits du fichier gold ;
- **filtered_sequences_with_ocr_context.csv** : séquences insérées / non-transcrites / mal transcrites filtrées selon le nombre de caractères ou un caractère particulier, avec leurs contextes gauches et droits du fichier gold.

**Usage :**

```sh
python3 analyse_errors.py -g GOLD -o OCR -e {inserted_sequences, missing_sequences, wrong_sequences, proportion} -d {length, char}
```

__options :__

| Option                                     | Description                                                                                    |
|--------------------------------------------|------------------------------------------------------------------------------------------------|
| `-g GOLD`, `--gold GOLD`                   | Chemin vers le fichier gold                                                                    |
| `-o OCR`, `--ocr OCR`                      | Chemin vers le fichier OCR                                                                     |
| `-e ERROR_TYPE`, `--error_type ERROR_TYPE` | Type d'erreur à analyser : inserted_sequences, missing_sequences, wrong_sequences, proportion  |
| `-d DETAIL`, `--detail DETAIL`             | Détail de l'erreur à analyser : length, char                                                   |


__Notes :__

- Le script commence par demander le nombre de caractères à prendre en compte pour les contextes gauche et droit des séquences à analyser. Il faut alors entrer un nombre entier. Si aucun nombre n'est renseigné, le nombre de caractères pour les contextes est fixé à 50 par défaut.
- La valeur `length` pour l'option `detail` permet de filtrer les séquences (insérées, manquantes ou mal transcrites) de la longueur souhaitée. Dans un premier temps, le script demande un seuil de comparaison. Il faut alors entrer un nombre correspondant à la longueur (en nombre de caractères des séquences que l'on souhaite étudier), par exemple 5. Le script demandera ensuite la longueur des séquences par rapport à ce seuil. Pour obtenir toutes les séquences dont la longueur est égale à ce seuil, entrer `eq`. Pour obtenir toutes les séquences dont la longueur est supérieure ou égale à ce seuil, entrer `sup`. Pour obtenir toutes les séquences dont la longueur est inférieure ou égale à ce seuil, entrer `inf`.
- La valeur `char` pour l'option `detail` permet d'obtenir les séquences (insérées, manquantes ou mal transcrites) composées d'un seul caractère. Dans le cas de l'analyse d'un caractère inséré, le script demandera le caractère inséré à étudier. Dans le cas de l'analyse d'un caractère non transcrit, le script demandera le caractère non transcrit à étudier. Dans le cas de l'analyse d'un caractère mal transcrit, le script demandera le caractère mal transcrit à analyser puis le caractère transcrit à la place. Si on entre un caractère mal transcrit sans caractère transcrit à la place, on obtient toutes les mauvaises transcriptions du caractère mal transcrit. Si on entre un caractère transcrit mais pas le caractère d'origine, on obtient tous les cas où le caractère a été transcrit à tort.



**Exemple :**

```sh
python3 analyse_errors.py -g ../1_Alignement/results/Textes_complets/Aligned_texts/RETAS/Tesseract/retas_gold_aligned_with_ocr_tesseract.txt -o ../1_Alignement/results/Textes_complets/Aligned_texts/RETAS/Tesseract/retas_ocr_tesseract_aligned_with_gold.txt -e inserted_sequences -d length
```