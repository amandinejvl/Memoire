import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import string
import re



def save_to_csv(df, nom_fichier):
    """
    Sauvegarde un DataFrame en fichier CSV.

    Args:
        df (DataFrame): DataFrame à sauvegarder.
        nom_fichier (str): Nom du fichier CSV à sauvegarder.

    Returns:
        None
    """
    df.to_csv(nom_fichier, index=False, encoding='utf-8')
    print(f"Le fichier '{nom_fichier}' a été sauvegardé avec succès.")



def filter_sequence_length(df: pd.DataFrame, op: str, n: int) -> pd.DataFrame:
    """
    Filtre les séquences selon leur longueur

    Args:
        df (DataFrame): DataFrame contenant les séquences avec leurs contextes.
        op (str): Opérateur de comparaison ('==', '>=', '<=').
        n (int): Longueur à comparer.

    Returns:
        pd.DataFrame: DataFrame filtré.
    """
    if op == 'eq':
        return df[df['Chaîne'].str.len() == n]
    elif op == 'sup':
        return df[df['Chaîne'].str.len() >= n]
    elif op == 'inf':
        return df[df['Chaîne'].str.len() <= n]



def filter_chars(df: pd.DataFrame, orig_char: str, tr_char: str) -> pd.DataFrame:
    """
    Filtre les séquences correspondant au caractère à analyser.

    Args:
        df (DataFrame): DataFrame contenant les séquences avec leurs contextes.
        orig_char (str): Caractère d'origine.
        tr_char (str): Caractère transcrit.

    Returns:
        pd.DataFrame: DataFrame filtré.
    """
    if tr_char == '' and orig_char != '':
        return df[df['Chaîne'] == orig_char]
    elif tr_char!= '' and orig_char == '':
        return df[df['Transcription'] == tr_char]
    else:
        return df[(df['Chaîne'] == orig_char) & (df['Transcription'] == tr_char)]



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes contenant uniquement des arobases dans un DataFrame.

    Args:
        df (DataFrame): DataFrame à nettoyer.

    Returns:
        pd.DataFrame: DataFrame nettoyé.
    """
    if df['Chaîne'].str.fullmatch(r'@+|', na=False).all():
        cleaned_df = df.drop('Chaîne', axis=1)
    elif df['Transcription'].str.fullmatch(r'@+|', na=False).all():
        cleaned_df = df.drop('Transcription', axis=1)
    else:
        cleaned_df = df
    return cleaned_df


def filter_user_length(sequences_with_gold_context: pd.DataFrame, sequences_with_ocr_context: pd.DataFrame) -> None:
    """
    Demande à l'utilisateur la longueur des séquences qu'il souhaite analyser et sauvegarde les résultats dans 2 fichiers csv.

    Args:
        sequences_with_gold_context (pd.DataFrame): DataFrame des séquences avec leur contexte gold.
        sequences_with_ocr_context (pd.DataFrame): DataFrame des séquences avec leur contexte ocr.

    Returns:
        None
    """
    # Demander un seuil de comparaison pour la longueur des séquences
    length = int(input("Seuil de comparaison pour la longueur des séquences : "))

    # Demander un opérateur de comparaison par rapport à la longueur choisie
    operator = input("Longueur des séquences par rapport à la longueur choisie ('eq', 'inf', 'sup') : ")

    # Filtrer les séquences selon la longueur choisie
    filtered_sequences_with_gold_context = filter_sequence_length(sequences_with_gold_context, operator, length)
    filtered_sequences_with_ocr_context = filter_sequence_length(sequences_with_ocr_context, operator, length)

    # Nettoyer les DataFrames en supprimant les colonnes contenant uniquement des arobases
    cleaned_filtered_sequences_with_gold_context = clean_dataframe(filtered_sequences_with_gold_context)
    cleaned_filtered_sequences_with_ocr_context = clean_dataframe(filtered_sequences_with_ocr_context)

    # Sauvegarder les résultats dans des fichiers csv
    save_to_csv(cleaned_filtered_sequences_with_gold_context, 'output/filtered_sequences_with_gold_context.csv')
    save_to_csv(cleaned_filtered_sequences_with_ocr_context, 'output/filtered_sequences_with_ocr_context.csv')



def filter_user_chars(sequences_with_gold_context: pd.DataFrame, sequences_with_ocr_context: pd.DataFrame, orig_char: str, tr_char: str) -> None:
    """
    Filtre les séquences en fonction du/des caractère(s) choisi(s) par l'utilisateur

    Args:
        sequences_with_gold_context (pd.DataFrame): DataFrame des séquences avec leur contexte gold.
        sequences_with_ocr_context (pd.DataFrame): DataFrame des séquences avec leur contexte ocr.
        orig_char (str): Caractère d'origine qui a été mal transcrit.
        tr_char (str): Caractère transcrit à la place du caractère d'origine.

    Returns:
        None
    """

    # Filtrer les séquences selon les caractères choisis
    filtered_sequences_with_gold_context = filter_chars(sequences_with_gold_context, orig_char, tr_char)
    filtered_sequences_with_ocr_context = filter_chars(sequences_with_ocr_context, orig_char, tr_char)

    # Nettoyer les DataFrames en supprimant les colonnes contenant uniquement des arobases
    cleaned_filtered_sequences_with_gold_context = clean_dataframe(filtered_sequences_with_gold_context)
    cleaned_filtered_sequences_with_ocr_context = clean_dataframe(filtered_sequences_with_ocr_context)

    # Sauvegarder les résultats dans des fichiers csv
    save_to_csv(cleaned_filtered_sequences_with_gold_context, 'output/filtered_sequences_with_gold_context.csv')
    save_to_csv(cleaned_filtered_sequences_with_ocr_context, 'output/filtered_sequences_with_ocr_context.csv')



def filter_char_according_to_error_type(sequences_with_gold_context: pd.DataFrame, sequences_with_ocr_context: pd.DataFrame, error_type: str) -> None:
    """
    Filtre les séquences par caractère en fonction du type d'erreur.

    Args:
        sequences_with_gold_context (pd.DataFrame): DataFrame des séquences avec leur contexte gold.
        sequences_with_ocr_context (pd.DataFrame): DataFrame des séquences avec leur contexte ocr.
        error_type (str): Type d'erreur (inserted_sequences, missing_sequences ou wrong_sequences).

    Returns:
        None
    """

    if error_type not in ['inserted_sequences', 'missing_sequences', 'wrong_sequences']:
        print("Type d'erreur non reconnu.")
        return

    else:
        if error_type == 'inserted_sequences':
            orig_char = ''
            tr_char = input("Caractère inséré : ")
        elif error_type =='missing_sequences':
            orig_char = input("Caractère manquant : ")
            tr_char = ''
        elif error_type == 'wrong_sequences':
            orig_char = input("Caractère d'origine : ")
            tr_char = input("Caractère transcrit : ")
        filter_user_chars(sequences_with_gold_context, sequences_with_ocr_context, orig_char, tr_char)



def filter_sequences(sequences_with_gold_context: pd.DataFrame, sequences_with_ocr_context: pd.DataFrame, detail: str, error_type: str) -> None:
    """
    Applique les filtres en fonction des options choisies par l'utilisateur.

    Args:
        sequences_with_gold_context (pd.DataFrame): DataFrame des séquences avec leur contexte gold.
        sequences_with_ocr_context (pd.DataFrame): DataFrame des séquences avec leur contexte ocr.
        detail (str): Détail des séquences à analyser ('length' ou 'char').
        error_type (str): Type d'erreur (inserted_sequences, missing_sequences ou wrong_sequences).

    Returns:
        None
    """
    # Si detail = 'length', filtrer les séquences selon leur longueur
    if detail == 'length':
        filter_user_length(sequences_with_gold_context, sequences_with_ocr_context)

    # Si detail = 'char', filtrer les séquences selon le/les caractère(s) que l'on souhaite analyser
    elif detail == 'char':
        filter_char_according_to_error_type(sequences_with_gold_context, sequences_with_ocr_context, error_type)

    # Sinon afficher un message d'erreur
    else:
        print("Détail de l'erreur non reconnu")
        return
