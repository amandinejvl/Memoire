import sys
import os
from multiprocessing import Pool
from tqdm import tqdm
from pymyers import MyersRealTime
import re
import nltk
import Levenshtein
import csv
import numpy as np
import argparse



try:
    # import pour l'appel depuis alignment.py
    from .process_texts import *  
except ImportError:
    # import pour l'appel direct de RETAS.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from process_texts import *



def create_sublists(lst: list, n: int) -> list:
    """
    Divise une liste en n parties égales (ou presque égales).

    Args:
        lst (list): Liste à diviser.
        n (int): Nombre de parties.
    
    Returns:
        list: Liste des sous-listes.
    """
    if n <= 0:
        raise ValueError("Le nombre de parties doit être supérieur à 0")
    
    size = len(lst)
    quotient, reste = divmod(size, n)

    result = []
    start = 0

    for i in range(n):
        end = start + quotient + (1 if i < reste else 0)
        result.append(lst[start:end])
        start = end

    return result



def create_sublists_with_margins(lst: list, n: int, margin: int) -> list:
    """
    Divise une liste en n parties égales (ou presque égales) avec une marge gauche et une marge droite.

    Args:
        lst (list): Liste à diviser.
        n (int): Nombre de parties.
        margin (int): Marge à ajouter à gauche et droite des parties.
    
    Returns:
        list: Liste des sous-listes avec marge.
    """
    if n <= 0:
        raise ValueError("Le nombre de parties doit être supérieur à 0")
    
    size = len(lst)
    quotient, reste = divmod(size, n)

    result = []
    start = 0

    for i in range(n):
        end = start + quotient + margin + (1 if i < reste else 0)
        if start != 0:
            start = start-margin
        result.append(lst[start:end])
        start = end-margin

    return result



def find_best_match(input_str: str, candidate_strings: list) -> tuple:
    """
    Calcule le meilleur match selon l'algorithme de Myers pour une chaîne d'entrée avec une liste de chaînes candidates.

    Args:
        input_str (str): Chaîne d'entrée à matcher.
        candidate_strings (list): Liste de chaînes candidates.
    
    Returns:
        tuple: Chaîne candidate la plus proche et nombre de matchs avec la chaîne d'entrée.
    """    
    # Initialiser le meilleur match et son indice à None et le nombre maximum de matches à 0
    best_match = None
    indice = None
    max_matches = 0
    seuil = 0.9
    
    # Comparer la chaîne d'entrée avec chaque chaîne candidate
    for i, candidate in enumerate(candidate_strings):
        try:
            myers = MyersRealTime(input_str, candidate)
            diff_re = myers.diff()

            # Calcul du taux normalisé de matches entre la phrase d'entrée et la phrase candidate
            nb_matches_normalised = len(diff_re.matches)/max(len(input_str), len(candidate))
            
            # Si le taux normalisé de matches est supérieur au seuil, mettre à jour le meilleur match et stopper l'itération
            if nb_matches_normalised > seuil:
                max_matches = len(diff_re.matches)
                best_match, indice = (candidate, i)
                break
            
            # Si le nombre de matches pour cette comparaison est plus élevé, mettre à jour le meilleur match
            elif len(diff_re.matches) > max_matches:
                max_matches = len(diff_re.matches)
                best_match, indice = (candidate, i)
            
            # Sinon, continuer l'itération pour les phrases suivantes
            else:
                continue
            
        except RecursionError:
            continue
    
    # Pour éviter de recomparer avec des phrases déjà associées par la suite
    if indice is not None:
        candidate_strings.pop(indice)

    return best_match, max_matches



def traiter_paire(paire: tuple) -> list:
    """
    Fonction qui traite une paire (gold_subliste, ocr_subliste) et renvoie la liste de résultats.
    
    Args:
        paire (tuple): Paire de segments (gold_subliste, ocr_subliste).
    
    Returns:
        list: Liste des résultats pour cette paire.
    """
    gold_sublist, ocr_sublist = paire
    resultats = []

    for gold_sent in gold_sublist:
        best_match, max_matches = find_best_match(gold_sent, ocr_sublist)

        print(gold_sent)
        print(best_match)
        print("_" * 80)
        if best_match is not None:
            cer_score = compute_cer(gold_sent, best_match)
            resultats.append([gold_sent, best_match, cer_score])
    return resultats



def MYERS(gold_text: str, ocr_text: str) -> None:
    """
    Aligne un texte OCR avec son gold en utilisant le Diff Algorithm de Myers.

    Args:
        gold_text (str): Texte gold.
        ocr_text (str): Texte OCR.
    
    Returns:
        None.
    """

    # Segmenter les textes en phrases
    gold_sentences = text_to_sentences(gold_text)
    ocr_sentences = text_to_sentences(ocr_text)
    print("Nombre de phrases gold :", len(gold_sentences))
    print("Nombre de phrases ocr  :", len(ocr_sentences))

    # Diviser les textes en sous-listes égales (avec une marge)
    nb_sublists = int(input("Entrez la valeur du nombre de sous-listes que vous souhaitez utiliser : "))
    margin = int(0.25 * max(len(gold_sentences), len(ocr_sentences)) // nb_sublists)
    print("Margin :", margin)
    gold_sublists = create_sublists(gold_sentences, nb_sublists)
    ocr_sublists = create_sublists_with_margins(ocr_sentences, nb_sublists, margin)

    # Préparation des paires de sous-listes
    paires = list(zip(gold_sublists, ocr_sublists))
    
    final_results = []
    # Utilisation de multiprocessing pour traiter chaque paire en parallèle
    with Pool() as pool:
        liste_resultats = list(tqdm(pool.imap(traiter_paire, paires), total=len(paires)))
    
    # Concaténer les résultats obtenus pour chaque paire
    for res in liste_resultats:
        final_results.extend(res)
    
    # Sauvegarder les résultats dans un fichier CSV
    save_to_csv(final_results, 'output/results_myers.csv')



if __name__ == '__main__':

    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Aligne un texte OCR avec son gold en utilisant le Diff Algorithm de Myers.')
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

    # Aligner les textes gold et OCR avec Myers
    MYERS(gold_text, ocr_text)