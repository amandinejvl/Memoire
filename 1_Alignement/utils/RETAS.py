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



try:
    # import pour l'appel depuis alignment.py
    from .process_texts import *  
except ImportError:
    # import pour l'appel direct de RETAS.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from process_texts import *



def RETAS(gold_text: str, ocr_text: str) -> None:
    """
    Aligne un texte OCR avec son gold en utilisant la méthode RETAS.

    Args:
        gold_text (str): Texte gold.
        ocr_text (str): Texte OCR.
    
    Returns:
        None.
    """

    # Aligner les textes gold et OCR avec RETAS
    aligned_gold, aligned_ocr = anchor.align_w_anchor(gold_text, ocr_text, gap_char="@")

    # Sauvegarder les textes alignés
    save_to_txt(aligned_gold, 'output/gold_aligned_with_ocr.txt')
    save_to_txt(aligned_ocr, 'output/ocr_aligned_with_gold.txt')

    # Segmenter le gold aligné en phrases et extraire les positions de chacune
    gold_sent_positions = text_to_sent_dict(aligned_gold)

    # Utiliser les positions pour trouver les phrases correspondantes dans l'OCR aligné
    aligned_sentences = find_corr_sent(aligned_ocr, gold_sent_positions)

    # Calculer le CER pour chaque paire de phrases
    results = []
    for gold_sent, ocr_sent in aligned_sentences.items():
        cer_score = compute_cer(gold_sent, ocr_sent)
        results.append([gold_sent, ocr_sent, cer_score])
    
    # Sauvegarder les résultats dans un fichier CSV
    save_to_csv(results, 'output/results_retas.csv')



if __name__ == "__main__":

    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Aligne un texte OCR avec son gold en utilisant la méthode RETAS.')
    parser.add_argument("-g", "--gold", type=str, help="Chemin vers le fichier gold")
    parser.add_argument("-o", "--ocr", type=str, help="Chemin vers le fichier OCR")
    args = parser.parse_args()

    # Vérification des arguments
    if not args.gold or not args.ocr:
        print("Veuillez fournir les chemins vers les fichiers gold et OCR.")
        exit()

    # Lire les textes gold et OCR
    gold_text = read_file(args.gold)
    ocr_text  = read_file(args.ocr)

    # Aligner les textes gold et OCR avec RETAS
    RETAS(gold_text, ocr_text)