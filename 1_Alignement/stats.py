import csv
import matplotlib.pyplot as plt
import numpy as np
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



def calculer_moyenne(valeurs: list) -> float:
    """
    Calcule et retourne la moyenne des valeurs d'une liste.
    
    Args:
        valeurs (list): Liste des valeurs à calculer la moyenne.
    
    Returns:
        float: La moyenne des valeurs.
    """
    moyenne = np.mean(valeurs)
    return moyenne



def afficher_distribution(liste_nombres: list) -> None:
    """
    Affiche un graphique représentant la distribution des nombres d'une liste.
    
    Args:
        liste_nombres (list): Liste des nombres à analyser.
    
    Returns:
        None.
    """
    intervalles_dict = {
        "0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1": 0,
        "supérieur à 1": 0
    }

    for nombre in liste_nombres:
        if 0 <= nombre < 0.2:
            intervalles_dict["0-0.2"] += 1
        elif 0.2 <= nombre < 0.4:
            intervalles_dict["0.2-0.4"] += 1
        elif 0.4 <= nombre < 0.6:
            intervalles_dict["0.4-0.6"] += 1
        elif 0.6 <= nombre < 0.8:
            intervalles_dict["0.6-0.8"] += 1
        elif 0.8 <= nombre < 1:
            intervalles_dict["0.8-1"] += 1
        elif nombre > 1:
            intervalles_dict["supérieur à 1"] += 1

    intervalles_dict_rel = {interval: count / len(liste_nombres) for interval, count in intervalles_dict.items()}
    intervalles = list(intervalles_dict_rel.keys())
    comptages = list(intervalles_dict_rel.values())

    plt.figure(figsize=(4, 5))
    plt.bar(intervalles, comptages, color='skyblue')
    plt.xlabel('Score de similarité')
    plt.ylabel('Fréquence')
    plt.title('Distribution des scores de similarité')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()



def analyse_results(algo_name: str, results_file: str) -> None:
    """
    Analyse les résultats de l'alignement réalisé par un algorithme.
    
    Args:
        algo_name (str): Nom de l'algorithme utilisé.
        results_file (str): Fichier CSV contenant les résultats de l'alignement.
    
    Returns:
        None.
    """
    algo_results = read_from_csv(results_file)
    if algo_results is None:
        return
    print(f"Nombre de phrases {algo_name}: {len(algo_results)}")
    algo_cers = [float(paire["CER"]) for paire in algo_results]
    moyenne_algo = calculer_moyenne(algo_cers)
    print(f"Moyenne des CER pour {algo_name}: {moyenne_algo}")
    afficher_distribution(algo_cers)



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Analyse les CER obtenus avec chaque algorithme d'alignement.")
    parser.add_argument("-a", "--algo", type=str, help="Algorithme d'alignement utilisé (myers, retas ou semantic)")
    parser.add_argument("-r", "--results", type=str, help="Chemin vers le fichier csv de résultats obtenus avec l'algorithme d'alignement choisi")
    args = parser.parse_args()

    # Vérification des arguments
    if not args.algo or not args.results:
        print("Veuillez fournir l'algorithme d'alignement et le fichier de résultats.")
        return
    if args.algo not in ['myers','retas', 'semantic']:
        print("Algorithme d'alignement inconnu. Choisissez myers, retas ou semantic.")
        return

    # Analyse des résultats pour l'algorithme et le système OCR choisis
    if args.algo == 'myers':
        analyse_results("Myers", args.results)
    elif args.algo == 'retas':
        analyse_results("RETAS", args.results)
    elif args.algo == 'semantic':
        analyse_results("Semantic Search", args.results)
    else:
        print("Algorithme inconnu. Choisissez 'myers', 'retas' ou 'semantic'.")
        return



if __name__ == "__main__":
    main()