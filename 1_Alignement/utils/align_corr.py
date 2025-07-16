import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'genalog')))
from genalog.text import alignment
from genalog.text import anchor
import Levenshtein
import re
import nltk
import csv
import argparse
nltk.download('punkt_tab')
import pandas as pd



try:
    # import pour l'appel depuis alignment.py
    from .process_texts import *  
except ImportError:
    # import pour l'appel direct de RETAS.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from process_texts import *



def extract_sentences(df: pd.DataFrame) -> list:
    """
    Extrait les phrases depuis le DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les phrases gold, ocr et corrigées.
    
    Returns:
        sentences (list): Liste des phrases gold, ocr, et corrigées.
    """

    # Extraction des phrases gold, ocr et corrigées
    gold_sentences = df['Ground Truth'].tolist()
    ocr_sentences = df['OCR Text'].tolist()
    corr_sentences = df['Model Correction'].tolist()

    return gold_sentences, ocr_sentences, corr_sentences



def align_gold_ocr_corr(gold_sentences: list, ocr_sentences: list, corr_sentences: list) -> list:
    """
    Aligne les phrases gold, ocr et correction.

    Args:
        gold_sentences (list): Liste des phrases gold.
        ocr_sentences (list): Liste des phrases ocr.
        corr_sentences (list): Liste des phrases correction.
    
    Returns:
        results (list): Liste des résultats des alignements.
    """

    # Initialiser une liste pour stocker les résultats
    results = []

    # Initialiser des variables pour vérifier les alignements
    val = True
    valid = 0
    invalid = 0

    # Parcourir chaque triplet de phrases
    for gold, ocr, corr in zip(gold_sentences, ocr_sentences, corr_sentences):

        # Aligner le gold et l'ocr
        GO, OG = anchor.align_w_anchor(gold, ocr, gap_char="@")

        # Aligner le gold et la correction
        GC, CG = anchor.align_w_anchor(gold, corr, gap_char="@")

        # Aligner les deux golds ensemble
        GO_GC, GC_GO = anchor.align_w_anchor(GO, GC, gap_char="@")

        # Aligner l'ocr aligné avec le gold avec le gold aligné avec l'ocr aligné avec l'autre gold
        OG_GOGC, GOGC_OG = anchor.align_w_anchor(OG, GO_GC, gap_char="@")

        # Aligner la correction alignée avec le gold avec le gold aligné avec l'ocr aligné avec l'autre gold
        CG_GOGC, GOGC_CG = anchor.align_w_anchor(CG, GO_GC, gap_char="@")

        # Vérifier si les alignements sont égaux sur toutes les lignes
        if len(GO_GC) == len(GC_GO) == len(OG_GOGC) == len(CG_GOGC):
            val = True
            valid += 1
        else:
            val = False
            invalid += 1

        # Sauvegarder la ligne dans la liste
        results.append((OG_GOGC, GO_GC, CG_GOGC, val))
    
    # Afficher le nombre de phrases alignées correctement et incorrectement
    print(f"Nombre de phrases bien alignées: {valid}")
    print(f"Nombre de phrases mal alignées: {invalid}")
    
    return results



def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description='Aligne les phrases sous forme de triplets (gold, ocr, correction) en utilisant la méthode RETAS.')
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv contenant les phrases originales avec leur transcription et leur correction")
    args = parser.parse_args()

    # Vérification de l'argument
    if not args.file:
        print("Veuillez fournir le chemin vers le fichier csv contenant les phrases originales avec leur transcription et leur correction.")
        exit()

    # Lire le fichier csv
    df = pd.read_csv(args.file)

    # Extraire les phrases gold, ocr et corrigées
    gold_sentences, ocr_sentences, corr_sentences = extract_sentences(df)

    # Aligner les triplets de phrases
    aligned_sentences = align_gold_ocr_corr(gold_sentences, ocr_sentences, corr_sentences)
    
    # Sauvegarder la liste des alignments dans un fichier csv
    df_results = pd.DataFrame(aligned_sentences, columns=['OCR Text', 'Ground Truth', 'Model Correction', 'Correct Alignment'])
    df_results.to_csv('output/aligned_corrections.csv', index=False)



if __name__ == "__main__":
    main()