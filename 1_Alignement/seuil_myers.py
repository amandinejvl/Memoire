import csv
import matplotlib.pyplot as plt
import numpy as np
from pymyers import MyersRealTime
import re
import nltk
import Levenshtein
from tqdm import tqdm
import statistics
import argparse



def read_from_csv(filename: str) -> list:
    """
    Lit les résultats de textes alignés depuis un fichier CSV et les retourne sous forme de liste de dictionnaires.
    
    Args:
        filename (str): Nom du fichier CSV.
    
    Returns:
        list: Liste des dictionnaires contenant les segments alignés.
    """
    results = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Ajouter chaque ligne du CSV sous forme de dictionnaire à la liste results
                results.append(row)
    except FileNotFoundError:
        print(f"Le fichier '{filename}' n'a pas été trouvé.")
        return None
    return results




def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description='Détermine le seuil à partir duquel on peut raisonnablement arrêter les comparaisons pour une phrase gold donnée.')
    parser.add_argument("-r", "--results", type=str, help="Chemin vers le fichier csv de résultats")
    args = parser.parse_args()

    # Vérification de l'argument
    if not args.results:
        print("Veuillez fournir un chemin vers le fichier csv de résultats.")
        return

    # Lire le fichier de résultats
    results_file = args.results
    results = read_from_csv(results_file)

    liste_nb_matches_normalised = []
    paires = [(paire["gold_sent"], paire["ocr_sent"]) for paire in results]

    for paire in tqdm(paires):
        try: 
            gold = paire[0]
            ocr = paire[1]

            myers = MyersRealTime(gold, ocr)
            diff_re = myers.diff()
            nb_matches_normalised = len(diff_re.matches)/max(len(gold), len(ocr))
            liste_nb_matches_normalised.append(nb_matches_normalised)
        
        except RecursionError:
            continue
    
    moyenne = np.mean(liste_nb_matches_normalised)
    print("Moyenne: ")
    print(moyenne)
    print("Minimum: ")
    print(min(liste_nb_matches_normalised))
    print("Maximum: ")
    print(max(liste_nb_matches_normalised))
    print("Ecart-type: ")
    print(statistics.stdev(liste_nb_matches_normalised))



if __name__ == "__main__":
    main()