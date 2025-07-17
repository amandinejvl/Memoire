import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import string
import argparse
from utils.find_sequences import *
from utils.filters import *



def get_positions_according_to_error_type(gold: str, ocr: str, error_type: str) -> list:
    """
    Trouve les positions des insertions/suppressions/erreurs selon le type d'erreur.

    Args:
        gold (str): Texte gold.
        ocr (str): Texte OCR.
        error_type (str): Type d'erreur à analyser ('inserted_sequences', 'missing_sequences', 'wrong_sequences').

    Returns:
        list: Liste des positions des insertions/suppressions/erreurs.
    """
    # Si on analyse des séquences insérées, trouver les positions des arobases dans le gold
    if error_type == 'inserted_sequences':
        positions = get_arobases_position(gold)
        return positions

    # Si on analyse des séquences manquantes, trouver les positions des arobases dans l'OCR
    elif error_type == 'missing_sequences':
        positions = get_arobases_position(ocr)
        return positions

    # Si on analyse des séquences mal transcrites, trouver les positions des caractères mal transcrits
    elif error_type == 'wrong_sequences':
        wrong_sequences = transcription_errors(gold, ocr)
        positions = [pos[0] for pos in wrong_sequences]
        return positions

    # Sinon afficher un message d'erreur
    else:
        print("Type d'erreur non reconnu")
        return



def process_sequences(sequences: list, transcribed_sequences: list, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Découpe chaque séquence en contexte en tuple (left_context, orig_seq, transcribed_seq, right_context)
    et enregistre :
    - dans un DataFrame avec les contextes du gold
    - un DataFrame avec les contextes de l'OCR.

    Args:
        sequences (list): Liste des séquences mal transcrites par le système OCR avec n caractères à gauche et à droite.
        transcribed_sequences (list): Liste des séquences transcrites par le système OCR avec n caractères à gauche et à droite.
        n (int): Nombre de caractères à considérer dans les contextes.

    Returns:
        tuple: Deux DataFrames avec les contextes du gold et les contextes de l'OCR.
    """
    # Initialiser une liste des séquences avec leurs contextes
    sequences_with_gold_context = []
    sequences_with_ocr_context = []

    # Découper chaque séquence en tuple (left_context, seq, tr_seq, right_context)
    for seq, tr_seq in zip(sequences, transcribed_sequences):
        left_gold_context, left_ocr_context = seq[:n], tr_seq[:n]
        orig_seq = seq[n:n+len(seq)-2*n]
        tr_orig_seq = tr_seq[n:n+len(seq)-2*n]
        right_gold_context, right_ocr_context = seq[n+len(seq)-2*n:], tr_seq[n+len(seq)-2*n:]
        sequences_with_gold_context.append((left_gold_context, orig_seq, tr_orig_seq, right_gold_context))
        sequences_with_ocr_context.append((left_ocr_context, orig_seq, tr_orig_seq, right_ocr_context))

    # Enregistrer les listes dans des DataFrames
    gold_df = pd.DataFrame(sequences_with_gold_context, columns=['Contexte gauche', 'Chaîne', 'Transcription', 'Contexte droit'])
    ocr_df = pd.DataFrame(sequences_with_ocr_context, columns=['Contexte gauche', 'Chaîne', 'Transcription', 'Contexte droit'])

    return gold_df, ocr_df



def get_sequences_with_context(gold: str, ocr: str, n:int, error_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trouve les séquences à analyser avec leur contexte.

    Args:
        gold (str): Texte gold.
        ocr (str): Texte OCR.
        n (int): Nombre de caractères à considérer dans les contextes.
        error_type (str): Type d'erreur (inserted_sequences, missing_sequences, wrong_sequences).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames des séquences avec leur contexte gold et ocr.
    """

    # Trouver les positions des insertions/suppressions/erreurs selon le type d'erreur
    positions = get_positions_according_to_error_type(gold, ocr, error_type)

    # Créer des intervalles à partir des positions
    intervals = create_intervals_from_positions(positions)

    # Elargir les intervalles à n caractères à gauche et à droite
    for i in range(len(intervals)):
        intervals[i] = (intervals[i][0] - n if intervals[i][0] - n >= 0 else 0, intervals[i][1] + n)

    # Retrouver les séquences textuelles correspondant aux intervalles dans le texte 2
    sequences = find_sequence_from_intervals(gold, intervals)
    transcribed_sequences = find_sequence_from_intervals(ocr, intervals)

    # Créer des concordanciers
    gold_df, ocr_df = process_sequences(sequences, transcribed_sequences, n)

    return gold_df, ocr_df



def check_args(gold: str, ocr: str, error_type: str, detail: str) -> bool:
    """
    Vérifie les arguments fournis par l'utilisateur.

    Args:
        gold (str): Chemin vers le fichier gold aligné avec l'OCR.
        ocr (str): Chemin vers le fichier OCR aligné avec le gold.
        error_type (str): Type d'erreur à analyser ('inserted_sequences','missing_sequences', 'wrong_sequences').
        detail (str): Détail de l'erreur à analyser ('length', 'char').

    Returns:
        bool: True si les arguments sont valides, False sinon.
    """

    # Partir du principe que les arguments sont valides
    valid_args = True

    # Vérifier gold et ocr
    if not gold or not ocr:
        print("Veuillez fournir les chemins vers les fichiers gold et OCR alignés ainsi que le type d'erreur que vous souhaitez analyser.")
        valid_args = False

    # Vérifier error_type
    if error_type not in ['inserted_sequences','missing_sequences', 'wrong_sequences']:
        print("Type d'erreur inconnu. Choisissez inserted_sequences, missing_sequences ou wrong_sequences.")
        valid_args = False

    # Vérifier detail
    if detail not in ['length', 'char']:
        print("Détail de l'erreur inconnu. Choisissez length ou char.")
        valid_args = False

    return valid_args



def ask_for_context_length(n: int) -> int:
    """
    Demande à l'utilisateur de choisir la longueur des contextes en nombre de caractères.

    Args:
        n (int): Longueur des contextes en nombre de caractères par défaut.

    Returns:
        int: Longueur des contextes en nombre de caractères.
    """
    context_length = input("Nombre de caractères pour les contextes (par défaut 50) : ")
    try:
        context_length = int(context_length)
    except ValueError:
        print(f"Le nombre entré n'est pas valide, les contextes seront initialisés à {n} caractères.")
        context_length = n
    return context_length



def analyse_sequences(gold: str, ocr: str, error_type: str, context_length: int, detail: str) -> None:
    """
    Analyse les séquences insérées / non-transcrites / mal transcrites par le système OCR

    Args:
        gold (str): Texte gold.
        ocr (str): Texte OCR.
        error_type (str): Type d'erreur (inserted_sequences, missing_sequences ou wrong_sequences).
        context_length (int): Longueur des contextes en nombre de caractères.

    Returns:
        None
    """
    # Extraire les séquences et les sauvegarder avec leur contexte
    sequences_with_gold_context, sequences_with_ocr_context = get_sequences_with_context(gold, ocr, context_length, error_type)

    # Nettoyer les DataFrames en supprimant les colonnes contenant uniquement des arobases
    cleaned_sequences_with_gold_context = clean_dataframe(sequences_with_gold_context)
    cleaned_sequences_with_ocr_context = clean_dataframe(sequences_with_ocr_context)

    # Sauvegarder les séquences avec leur contexte dans des fichiers csv
    save_to_csv(cleaned_sequences_with_gold_context, 'output/sequences_with_gold_context.csv')
    save_to_csv(cleaned_sequences_with_ocr_context, 'output/sequences_with_ocr_context.csv')

    # Filtrer les séquences en fonction des options choisies par l'utilisateur
    filter_sequences(sequences_with_gold_context, sequences_with_ocr_context, detail, error_type)



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Analyse les erreurs produites par le système OCR.')
    parser.add_argument("-g", "--gold", type=str, help="Chemin vers le fichier gold aligné avec l'OCR")
    parser.add_argument("-o", "--ocr", type=str, help="Chemin vers le fichier OCR aligné avec le gold")
    parser.add_argument("-e", "--error_type", type=str, help="Type d'erreur à analyser (inserted_sequences, missing_sequences, wrong_sequences)")
    parser.add_argument("-d", "--detail", type=str, help="Détail de l'erreur à analyser (length, char)")
    args = parser.parse_args()

    # Vérification des arguments
    valid = check_args(args.gold, args.ocr, args.error_type, args.detail)
    if valid == False:
        return

    # Demander à l'utilisateur le nombre de caractères qu'il veut pour les contextes (par défaut 50)
    context_length = ask_for_context_length(50)

    # Lire les fichiers gold et ocr alignés
    gold_text = read_file(args.gold)
    ocr_text  = read_file(args.ocr)

    # Vérifier si les deux fichiers ont été correctement chargés
    if gold_text is None or ocr_text is None:
        return

    # Analyser les erreurs
    if args.error_type not in ['inserted_sequences', 'missing_sequences', 'wrong_sequences']:
        print("Type d'erreur inconnu. Choisissez inserted_sequences, missing_sequences ou wrong_sequences.")
        return
    else:
        print(f"Analyse of {args.error_type.replace('_', ' ')}")
        analyse_sequences(gold_text, ocr_text, args.error_type, context_length, args.detail)



if __name__ == "__main__":
    main()
