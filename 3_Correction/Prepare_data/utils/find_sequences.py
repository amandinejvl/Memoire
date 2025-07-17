import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import Levenshtein



def get_arobases_position(text: str) -> list:
    """
    Retourne les positions des arobases dans le texte.

    Args:
        text (str): Texte à analyser.
    
    Returns:
        list: Liste des positions des arobases dans le texte.
    """
    positions = []
    for i, char in enumerate(text):
        if char == "@":
            positions.append(i)
    return positions



def get_maj_position(text: str) -> list:
    """
    Retourne les positions des majuscules dans le texte.

    Args:
        text (str): Texte à analyser.
    
    Returns:
        list: Liste des positions des majuscules dans le texte.
    """
    positions = []
    for i, char in enumerate(text):
        if char.isupper():
            positions.append(i)
    return positions



def create_intervals_from_positions(liste_positions: list) -> list:
    """
    Crée des intervalles à partir d'une liste de positions.
    
    Args:
        liste_positions (list): Liste des positions.
    
    Returns:
        list: Liste des intervalles créés à partir des positions.
    """
    if not liste_positions:
        return []
    
    result = []
    start = liste_positions[0]

    for i in range(1, len(liste_positions)):
        if liste_positions[i] - liste_positions[i - 1] > 1:
            result.append((start, liste_positions[i - 1]) if start != liste_positions[i - 1] else (start, start))
            start = liste_positions[i]
    
    result.append((start, liste_positions[-1]) if start != liste_positions[-1] else (start, start))
    return result



def find_sequence_from_intervals(text: str, intervals: list) -> list:
    """
    Retrouve les séquences textuelles correspondant aux intervalles.

    Args:
        text (str): Texte dans lequel rechercher les séquences.
        intervals (list): Liste des intervalles.
    
    Returns:
        list: Liste des séquences textuelles correspondant aux intervalles.
    """
    result = []
    for interval in intervals:
        sequence = text[interval[0]:interval[1]+1]
        result.append(sequence)
    return result