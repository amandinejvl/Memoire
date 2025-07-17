import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import string
import argparse
from utils.find_sequences import *
from utils.plot import *



def compare_arobases(text1: str, text2: str) -> list:
    """
    Trouve les séquences insérées par le système OCR

    Args:
        text1 (str): Texte 1.
        text2 (str): Texte 2.
    
    Returns:
        list: Liste des séquences insérées par le système OCR.
    """
    # Trouver les positions des arobases dans le texte 1
    insertions_position = get_arobases_position(text1)

    # Créer des intervalles à partir des positions des arobases
    intervals = create_intervals_from_positions(insertions_position)

    # Retrouver les séquences textuelles correspondant aux intervalles dans le texte 2
    sequences = find_sequence_from_intervals(text2, intervals)

    return sequences



def regroup_errors(errors: list) -> list:
    """
    Regroupe les erreurs aux positions adjacentes.

    Args:
        errors (list): Liste des tuples (position, gold, ocr).
    
    Returns:
        list: Liste des groupes d'erreurs.
    """
    # Initialiser la liste pour stocker les groupes d'erreurs
    groups = []
    
    # Initialiser des listes temporaires pour suivre les positions, valeurs gold et ocr du groupe actuel
    current_positions = []
    current_gold = []
    current_ocr = []
    
    # Parcourir la liste des erreurs
    for i, (pos, gold, ocr) in enumerate(errors):
        
        # Vérifier si la position actuelle est adjacente à la précédente
        if current_positions and pos != current_positions[-1] + 1:
            # Si ce n'est pas le cas, ajouter le groupe précédent à la liste finale
            if len(current_positions) > 1:
                groups.append((f"{current_positions[0]}-{current_positions[-1]}", "".join(current_gold), "".join(current_ocr)))
            else:
                groups.append((str(current_positions[0]), current_gold[0], current_ocr[0]))
            
            # Réinitialiser les listes pour le prochain groupe
            current_positions = []
            current_gold = []
            current_ocr = []
        
        # Ajouter la position, gold et ocr au groupe en cours
        current_positions.append(pos)
        current_gold.append(gold)
        current_ocr.append(ocr)
    
    # Ajouter le dernier groupe formé
    if current_positions:
        if len(current_positions) > 1:
            groups.append((f"{current_positions[0]}-{current_positions[-1]}", "".join(current_gold), "".join(current_ocr)))
        else:
            groups.append((str(current_positions[0]), current_gold[0], current_ocr[0]))
    
    return groups



def count_characters(liste: list) -> int:
    """
    Compte les caractères en fonction du type des éléments de la liste.

    Args:
        liste (list): Liste des éléments.
    
    Returns:
        int: Le nombre de caractères.
    """
    # Si le type des éléments est une liste de tuples (position, gold, ocr)
    if type(liste[0]) == tuple:
        nb_chars = len(liste)
    # Si le type des éléments est une liste de chaînes de caractères
    elif type(liste[0]) == str:
        nb_chars = 0
        for seq in liste:
            nb_chars += len(seq)
    else:
        raise ValueError("Les éléments de la liste doivent être de types tuples ou de chaînes de caractères.")
    return nb_chars



def analyse_error(sequences: list) -> None:
    """
    Analyse les séquences d'entrée.

    Args:
        sequences (list): Liste des séquences.
    
    Returns:
        None
    """
    # Afficher le nombre de séquences insérées/non-transcrites
    print(f'{len(sequences)} séquences')

    # Afficher le nombre de caractères insérés/non-transcrits
    nb_chars = count_characters(sequences)
    print(f'{nb_chars} caractères')

    # Afficher la distribution des longueurs de séquences insérées/non-transcrites
    plot_length_distribution(sequences)

    # Afficher la distribution des caractères uniques insérés/non-transcrits
    plot_characters_distribution(sequences)

    # Afficher la distribution des types de caractères uniques insérés/non-transcrits
    plot_type_characters_distribution(sequences)



def analyse_wrong_transcriptions(errors: list) -> None:
    """
    Analyse les séquences mal transcrites par le système OCR.

    Args:
        errors (list): Liste des tuples (position, gold, ocr).
    
    Returns:
        None
    """
    # Regrouper les erreurs aux positions adjacentes
    grouped_errors = regroup_errors(errors)

    # Récupérer les erreurs caractère par caractère
    unique_char_errors = [(error[1], error[2]) for error in errors]

    # Récupérer les séquences mal transcrites (ocr)
    wrong_sequences = [error[2] for error in grouped_errors]

    # Afficher le nombre de séquences mal transcrites
    print(f'{len(wrong_sequences)} séquences')

    # Afficher le nombre de caractères mal transcrits
    print(f'{len(errors)} caractères')

    # Afficher la distribution des longueurs de séquences mal transcrites
    plot_length_distribution(wrong_sequences)

    # Tracer une heatmap des erreurs de transcription OCR
    plot_ocr_heatmap(unique_char_errors)

    # Afficher la distribution des caractères uniques mal reconnus
    plot_characters_distribution([error[1] for error in errors])

    # Afficher la distribution des caractères uniques transcrits à tort
    plot_characters_distribution([error[2] for error in errors])

    # Afficher la distribution des types de caractères uniques mal transcrits
    plot_type_characters_distribution([error[1] for error in errors])

    # Afficher la distribution des types de caractères uniques transcrits à tort
    plot_type_characters_distribution([error[2] for error in errors])



def proportion_errors(error_type_1: list, error_type_2: list, error_type_3: list) -> tuple:
    """
    Calcule la proportion de chaque type d'erreur.

    Args:
        error_type_1 (list): Liste des erreurs de type 1.
        error_type_2 (list): Liste des erreurs de type 2.
        error_type_3 (list): Liste des erreurs de type 3.
    
    Returns:
        tuple: Proportions des erreurs pour chaque type.
    """
    # Compter le nombre de caractères pour chaque type d'erreur
    nb_chars_type_1 = count_characters(error_type_1)
    nb_chars_type_2 = count_characters(error_type_2)
    nb_chars_type_3 = count_characters(error_type_3)

    # Calculer le nombre total de caractères
    total = nb_chars_type_1 + nb_chars_type_2 + nb_chars_type_3

    # Calculer la proportion pour chaque type d'erreur
    proportion_type_1 = nb_chars_type_1 / total * 100
    proportion_type_2 = nb_chars_type_2 / total * 100
    proportion_type_3 = nb_chars_type_3 / total * 100

    return proportion_type_1, proportion_type_2, proportion_type_3



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Identifie les erreurs produites par le système OCR.')
    parser.add_argument("-g", "--gold", type=str, help="Chemin vers le fichier gold aligné avec l'OCR")
    parser.add_argument("-o", "--ocr", type=str, help="Chemin vers le fichier OCR aligné avec le gold")
    parser.add_argument("-e", "--error_type", type=str, help="Type d'erreur à analyser (inserted_sequences, missing_sequences, wrong_sequences, proportion)")
    args = parser.parse_args()

    # Vérification des arguments
    if not args.gold or not args.ocr:
        print("Veuillez fournir les chemins vers les fichiers gold et OCR alignés ainsi que le type d'erreur que vous souhaitez analyser.")
        return
    if args.error_type not in ['inserted_sequences','missing_sequences', 'wrong_sequences', 'proportion']:
        print("Type d'erreur inconnu. Choisissez inserted_sequences, missing_sequences, wrong_sequences ou proportion.")
        return

    # Lire les fichiers gold et ocr alignés
    gold_text = read_file(args.gold)
    ocr_text  = read_file(args.ocr)

    # Vérifier si les deux fichiers ont été correctement chargés
    if gold_text is None or ocr_text is None:
        return

    # Détecter les différents types d'erreurs
    inserted_sequences = compare_arobases(gold_text, ocr_text)
    missing_sequences = compare_arobases(ocr_text, gold_text)
    wrong_sequences = transcription_errors(gold_text, ocr_text)

    # Calculer la proportion de chaque erreur
    proportion_of_errors = proportion_errors(inserted_sequences, missing_sequences, wrong_sequences)

    # Analyser les erreurs
    if args.error_type == 'inserted_sequences':
        print("Analyse des séquences insérées :")
        analyse_error(inserted_sequences)
    elif args.error_type == 'missing_sequences':
        print("Analyse des séquences non transcrites :")
        analyse_error(missing_sequences)
    elif args.error_type == 'wrong_sequences':
        print("Analyse des séquences mal transcrites :")
        analyse_wrong_transcriptions(wrong_sequences)
    elif args.error_type == 'proportion':
        print(f"Proportion des erreurs : \n- caractères insérés : {proportion_of_errors[0]:.2f}% \n- caractères non transcrits : {proportion_of_errors[1]:.2f}% \n- caractères mal transcrits : {proportion_of_errors[2]:.2f}%")
        plot_proportions(proportion_of_errors)
    else:
        print("Type d'erreur inconnu. Choisissez inserted_sequences, missing_sequences, wrong_sequences ou proportion.")
    


if __name__ == "__main__":
    main()