import csv
import argparse
import Levenshtein
from utils.find_sequences import *



def read_from_csv(filename: str) -> list:
    """
    Lit les résultats de textes alignés depuis un fichier CSV et les retourne sous forme de liste de dictionnaires.
    
    Args:
        filename (str): Nom du fichier CSV.
    
    Returns:
        list: Liste des dictionnaires contenant les phrases alignées.
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



def save_to_csv(results: list, filename: str) -> None:
    """
    Sauvegarde les données nettoyées dans un fichier CSV et renomme les colonnes.
    
    Args:
        results (list): Liste des lignes à enregistrer, sous forme de tuples (id, gold_sent, ocr_sent).
        filename (str): Nom du fichier CSV.
    
    Returns:
        None.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "Ground Truth", "OCR Text", "CER"])
        writer.writerows(results)
    print(f"Les résultats ont bien été enregistrés dans le fichier {filename}.")




def cer(prediction: str, target: str) -> float:
    """
    Calcule le taux d'erreur par caractère entre la prédiction et la cible.

    Args:
        prediction (str): Texte correspondant à la prédiction (de l'OCR ou du modèle de correction).
        target (str): Texte cible (gold).

    Returns:
        float: CER entre la prédiction et la cible.
    """
    distance = Levenshtein.distance(prediction, target)
    return distance / len(target)



def delete_headers(lines: list) -> list:
    """
    Supprime les séquences de 10 arobases ou plus (en-têtes) et recalcule le CER.
    
    Args:
        lines (list): Liste des dictionnaires contenant les phrases alignées.
    
    Returns:
        list: Liste des dictionnaires contenant les phrases alignées normalisées.
    """

    # Initialiser une liste pour stocker les lignes normalisées
    lines_without_headers = []

    # Parcourir l'ensemble des paires de phrases alignées
    for line in lines:

        # Trouver les positions des arobases dans le gold
        positions = get_arobases_position(line['gold_sent'])

        # Créer des intervalles à partir des positions des arobases
        intervals = create_intervals_from_positions(positions)

        # Retrouver les séquences textuelles correspondant aux intervalles
        gold_sequences = find_sequence_from_intervals(line['gold_sent'], intervals)
        ocr_sequences = find_sequence_from_intervals(line['ocr_sent'], intervals)

        # Supprimer les séquences de 10 arobases ou plus
        for gold_seq, ocr_seq in zip(gold_sequences, ocr_sequences):
            if len(gold_seq) >= 10:
                line['gold_sent'] = line['gold_sent'].replace(gold_seq, "")
                line['ocr_sent'] = line['ocr_sent'].replace(ocr_seq, "")

        # Ajouter la ligne normalisée à la liste des résultats
        lines_without_headers.append(line)
    
    return lines_without_headers



def normalise_titles(lines: list) -> list:
    """
    Normalise les titres de chapitres (minuscule/majuscule) dans le texte aligné.

    Args:
        lines (list): Liste des dictionnaires contenant les phrases alignées.
    
    Returns:
        list: Liste des dictionnaires contenant les phrases alignées normalisées.
    """

    # Initialiser une liste pour stocker les lignes normalisées
    lines_with_normalised_titles = []

    # Parcourir l'ensemble des paires de phrases alignées
    for line in lines:

        # Trouver les positions des majuscules dans l'OCR
        positions = get_maj_position(line['ocr_sent'])

        # Créer des intervalles à partir des positions des majuscules
        intervals = create_intervals_from_positions(positions)

        # Retrouver les séquences textuelles correspondant aux intervalles
        gold_sequences = find_sequence_from_intervals(line['gold_sent'], intervals)
        ocr_sequences = find_sequence_from_intervals(line['ocr_sent'], intervals)

        # Mettre en majuscule dans le gold les séquences de 2 minuscules ou plus
        for gold_seq, ocr_seq in zip(gold_sequences, ocr_sequences):
            if len(gold_seq) >= 2 and gold_seq.upper() == ocr_seq :
                line['gold_sent'] = line['gold_sent'].replace(gold_seq, gold_seq.upper())
        
        # Ajouter la ligne normalisée à la liste des résultats
        lines_with_normalised_titles.append(line)
    
    return lines_with_normalised_titles



def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description='Supprime les arobases des phrases alignées et renomme les colonnes du csv.')
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier contenant les paires de phrases alignées.")
    args = parser.parse_args()

    # Vérification de l'argument
    if not args.file:
        print("Veuillez fournir le chemin vers le fichier contenant les paires de phrases alignées.")
        exit()

    # Lire le fichier csv
    aligned_data = read_from_csv(args.file)

    # Supprimer les séquences insérées correspondant à des infos présentes dans le texte numérisé mais pas dans le gold
    data_without_headers = delete_headers(aligned_data)

    # Normaliser les titres de chapitres (minuscule/majuscule)
    normalised_data = normalise_titles(data_without_headers)

    # Supprimer les arobases des phrases alignées et ajouter une colonne id
    cleaned_lines = []
    for index, line in enumerate(normalised_data):

        # Recalculer les CER
        line['CER'] = cer(line['ocr_sent'], line['gold_sent'])

        # Conserver uniquement les paires de phrases ayant un CER inférieur à 0.8
        if float(line['CER']) < 0.8:

            # Tronquer les phrases supérieures à 1024 caractères
            new_line = (index+1, line['gold_sent'][0:1024].replace("@", ""), line['ocr_sent'][0:1024].replace("@", ""), line['CER'])
            cleaned_lines.append(new_line)

    # Sauvegarder les données nettoyées dans un nouveau fichier CSV
    save_to_csv(cleaned_lines, "output/cleaned_aligned_lines.csv")



if __name__ == "__main__":
    main()