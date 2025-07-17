from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import csv



def check_args(file: str, test_size: str) -> bool:
    """
    Vérifie les arguments fournis par l'utilisateur.

    Args:
        file (str): Chemin vers le fichier csv contenant les paires de phrases alignées.
        test_size (str): Taille des données à conserver pour le test.

    Returns:
        bool: True si les arguments sont valides, False sinon.
    """
    # Partir du principe que les arguments sont valides
    valid_args = True

    # Vérifier file
    if not file:
        print("Veuillez fournir le chemin vers le fichier csv sur lequel vous souhaitez séparer les données.")
        valid_args = False

    # Vérifier test_size
    try:
        test_size = float(test_size)
    except ValueError:
        print("Veuillez fournir une taille de données valide pour le test (ex : 0.2 pour 20%).")
        valid_args = False

    return valid_args



def save_to_csv(ocr_sentences: list, gold_sentences: list, filename: str) -> None:
    """
    Sauvegarde les données X et y dans un nouveau fichier csv.

    Args:
        X (list): Liste des phrases OCR.
        y (list): Liste des phrases gold.
        filename (str): Chemin vers le fichier csv où sauvegarder les données.
    
    Returns:
        None
    """    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['OCR Text', 'Ground Truth'])
        for ocr_sent, gold_sent in zip(ocr_sentences, gold_sentences):
            writer.writerow([ocr_sent, gold_sent])



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Sépare les données pour l'entraînement d'un modèle.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv contenant les paires de phrases alignées")
    parser.add_argument("-s", "--test_size", type=str, help="Taille des données à conserver pour le test")
    args = parser.parse_args()

    # Vérification des arguments
    valid = check_args(args.file, args.test_size)
    if valid == False:
        return
    
    # Chargement du fichier de données
    df = pd.read_csv(args.file)

    # Séparation des colonnes OCR Text et Ground Truth
    X = df['OCR Text']
    y = df['Ground Truth']

    # Séparation des données en données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=float(args.test_size), shuffle=True)

    # Sauvegarde des données d'entrainement et de test dans un nouveau fichier csv
    save_to_csv(X_train, y_train, 'output/train.csv')
    save_to_csv(X_test, y_test, 'output/test.csv')



if __name__ == "__main__":
    main()