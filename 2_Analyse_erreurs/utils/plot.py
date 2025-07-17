import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import string



def plot_proportions(proportions: tuple) -> None:
    """
    Trace un graphique en barres représentant les proportions.

    Args:
        proportions (tuple): Proportions des erreurs pour chaque type.
    
    Returns:
        None
    """
    labels = ['Inserted', 'Missing', 'Wrong']
    y_pos = np.arange(len(labels))
    plt.figure(figsize=(4, 5))
    plt.bar(y_pos, proportions, align='center', alpha=0.5)
    # Ajout des valeurs au-dessus des barres
    for i, v in enumerate(proportions):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=8)
    plt.xticks(y_pos, labels)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylabel('Proportion (%)')
    plt.title('Proportion des erreurs par type')
    plt.show()



def plot_length_distribution(liste: list) -> None:
    """
    Trace un histogramme de la distribution des longueurs de séquences.

    Args:
        liste (list): Liste des séquences.
    
    Returns:
        None
    """
    # Calculer les longueurs des séquences à partir de la liste
    lengths = [len(sequence) for sequence in liste]

    # Identifier le seuil à partir duquel on met ensemble des valeurs
    #seuil = np.mean(lengths) + (max(lengths) / np.std(lengths))
    seuil = 10

    # Créer une nouvelle liste en prenant en compte le seuil
    lengths_for_visualisation = []
    for length in lengths: 
        if length >= seuil:
            new_length = int(seuil)
        else:
            new_length = length
        lengths_for_visualisation.append(new_length)

    # Créer un dictionnaire avec les longueurs des séquences et leurs fréquences
    intervalles_dict = {}
    for length in lengths_for_visualisation:
        if length in intervalles_dict:
            intervalles_dict[length] += 1
        else:
            intervalles_dict[length] = 1

    # Ajouter les entrées pour les valeurs pour lesquelles il n'y a pas d'occurrences
    for i in range(1, seuil+1):
        if i not in intervalles_dict:
            intervalles_dict[i] = 0

    # Trier les intervalles par longueur croissante
    intervalles_dict = dict(sorted(intervalles_dict.items(), key=lambda item: item[0]))

    # Normaliser la fréquence pour chaque intervalle
    for length, count in intervalles_dict.items():
        intervalles_dict[length] = count / len(lengths_for_visualisation)

    # Créer les listes pour les abscisses et les ordonnées
    intervalles = list(intervalles_dict.keys())
    comptages = list(intervalles_dict.values())

    # Tracer l'histogramme
    plt.figure(figsize=(4, 5))
    bars = plt.bar(intervalles, comptages, color='skyblue')

    # Ajouter les valeurs arrondies au-dessus de chaque barre
    for bar, freq in zip(bars, comptages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f"{round(freq, 2)}", ha='center', fontsize=7)

    # Personnalisation des axes
    plt.xlabel('Longueur des séquences')
    plt.ylabel('Fréquence')
    plt.title('Distribution des longueurs de séquences')

    # Modification des étiquettes des abscisses
    labels = [str(i) for i in intervalles]
    labels[-1] = f"{intervalles[-1]} et plus"
    plt.xticks(intervalles, labels, rotation=45)

    plt.tight_layout()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()



def plot_characters_distribution(liste: list) -> None:
    """
    Trace un histogramme de la distribution des caractères.
    
    Args:
        liste (list): Liste des séquences.
    
    Returns:
        None
    """
    # Extraire les caractères uniques
    unique_chars = []
    for seq in liste:
        if len(seq) == 1:
            if len(seq.strip()) == 0:
                new_seq = "#"
            else: 
                new_seq = seq
            unique_chars.append(new_seq)
    
    # Créer un dictionnaire pour regrouper les caractères identiques
    unique_chars_dict = {}
    for char in unique_chars:
        if char in unique_chars_dict:
            unique_chars_dict[char] += 1
        else:
            unique_chars_dict[char] = 1
    
    # Trier les caractères par fréquence décroissante
    unique_chars_dict = dict(sorted(unique_chars_dict.items(), key=lambda item: item[1], reverse=True))

    # Normaliser la fréquence pour chaque intervalle
    for char, count in unique_chars_dict.items():
        unique_chars_dict[char] = count / len(unique_chars)
        
    # Créer les listes pour les abscisses et les ordonnées
    chars = list(unique_chars_dict.keys())
    counts = list(unique_chars_dict.values())

    # Tracer l'histogramme en barres avec des labels adaptés au nombre de séquences
    plt.figure(figsize=(10, 6))
    plt.bar(chars, counts, color='skyblue')
    plt.xlabel('Caractères uniques')
    plt.ylabel('Fréquence')
    plt.title('Distribution des caractères uniques')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()



def plot_type_characters_distribution(liste: list) -> None:
    """
    Trace un histogramme de la distribution des types caractères.
    
    Args:
        liste (list): Liste des séquences.
    
    Returns:
        None
    """
    # Extraire les caractères uniques
    unique_chars = []
    for seq in liste:
        if len(seq) == 1:
            if len(seq.strip()) == 0:
                new_seq = "#"
            else: 
                new_seq = seq
            unique_chars.append(new_seq)

    # Regrouper les caractères uniques en 3 sous-groupes
    espaces = []
    ponctuations = []
    alphanum = []

    for char in unique_chars:
        if char == "#":
            espaces.append(char)
        elif char in string.punctuation:
            ponctuations.append(char)
        else:
            alphanum.append(char)

    # Créer un dictionnaire à partir des 3 sous-groupes
    groupes = {
        "Espaces": len(espaces)/len(unique_chars),
        "Ponctuations": len(ponctuations)/len(unique_chars),
        "Alphanumériques": len(alphanum)/len(unique_chars)
    }

    # Créer les listes pour les abscisses et les ordonnées
    chars = list(groupes.keys())
    counts = list(groupes.values())

    # Tracer l'histogramme
    plt.figure(figsize=(4, 5))
    bars = plt.bar(chars, counts, color='skyblue')

    # Ajouter les valeurs au-dessus des barres
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
                 f"{count:.2f}", ha='center', fontsize=10)

    plt.xlabel('Types de caractères')
    plt.ylabel('Fréquence')
    plt.title('Distribution des types de caractères')
    plt.xticks(fontsize=8)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()



def plot_ocr_heatmap(error_list: list) -> None:
    """
    Crée une heatmap des erreurs de transcription OCR.

    Args:
        error_list (list): Liste des tuples (gold, ocr).
    
    Returns:
        None
    """    
    error_counts = Counter(error_list)
    
    # Extraire les caractères uniques
    gold_chars = sorted(set(gold_char for gold_char, _ in error_counts))
    ocr_chars = sorted(set(ocr_char for _, ocr_char in error_counts))
    
    # Créer une matrice de confusion
    confusion_matrix = np.zeros((len(gold_chars), len(ocr_chars)))
    for (gold_char, ocr_char), count in error_counts.items():
        i = gold_chars.index(gold_char)
        j = ocr_chars.index(ocr_char)
        confusion_matrix[i, j] = count
    
    # Trouver les lignes et colonnes contenant au moins un élément > 10
    rows_to_keep = np.any(confusion_matrix > 10, axis=1)
    cols_to_keep = np.any(confusion_matrix > 10, axis=0)

    # Filtrer la matrice et les caractères correspondants
    filtered_matrix = confusion_matrix[rows_to_keep][:, cols_to_keep]
    filtered_gold_chars = np.array(gold_chars)[rows_to_keep].tolist()
    filtered_ocr_chars = np.array(ocr_chars)[cols_to_keep].tolist()

    # Créer un dataframe de la matrice de confusion pour seaborn
    df = pd.DataFrame(confusion_matrix, index=gold_chars, columns=ocr_chars)
    df.to_csv('char_confusions.csv')

    # Créer un dataframe de la matrice de confusion réduite pour seaborn
    df_reduced = pd.DataFrame(filtered_matrix, index=filtered_gold_chars, columns=filtered_ocr_chars)
    
    # Tracer la heatmap globale
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, fmt="g", cmap="Reds", linewidths=0.5)
    plt.xlabel("Caractères transcrits par l'OCR")
    plt.ylabel("Caractères du gold")
    plt.title("Heatmap des erreurs de transcription OCR")
    plt.show()

    # Tracer la heatmap réduite
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_reduced, annot=False, fmt="g", cmap="Reds", linewidths=0.5)
    plt.xlabel("Caractères transcrits par l'OCR")
    plt.ylabel("Caractères du gold")
    plt.title("Heatmap des erreurs de transcription OCR les plus fréquentes")
    plt.show()
