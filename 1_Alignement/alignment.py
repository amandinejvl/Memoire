import argparse
from utils.RETAS import RETAS
from utils.Myers import MYERS
from utils.semantic_search import SEMANTIC



def read_file(filename: str) -> str:
    """
    Lit le contenu d'un fichier et remplace les retours à la ligne par un espace sauf quand il est précédé d'un espace (mot tronqué par souci de mise en page)

    Args:
        filename (str): Nom du fichier.
    
    Returns:
        str: Contenu du fichier.
    """
    try:
        with open(filename, 'r') as file:
            content = file.read()
            content_without_retours = content.replace("-\n", "").replace("\n", " ")
            #final_content = re.sub(r"(?=[.,;:!?])", " ", content_without_retours)
            return content_without_retours
    except FileNotFoundError:
        print(f"The file '{filename}' was not found.")
        return None



def check_args(algo: str, gold: str, ocr: str) -> bool:
    """
    Vérifie les arguments fournis par l'utilisateur.

    Args:
        algo (str): Type d'algorithme d'alignement (myers, retas, semantic).
        gold (str): Chemin vers le fichier gold aligné avec l'OCR.
        ocr (str): Chemin vers le fichier OCR aligné avec le gold.

    Returns:
        bool: True si les arguments sont valides, False sinon.
    """

    # Partir du principe que les arguments sont valides
    valid_args = True

    # Verifier l'algorithme d'alignement choisi
    if algo not in ['myers','retas','semantic']:
        print("Algorithme d'alignement inconnu. Choisissez myers, retas ou semantic.")
        valid_args = False

    # Vérifier gold et ocr
    if not gold or not ocr:
        print("Veuillez fournir les chemins vers les fichiers gold et OCR alignés ainsi que le type d'erreur que vous souhaitez analyser.")
        valid_args = False

    return valid_args



def align_texts(algo: str, gold: str, ocr: str) -> None:
    """
    Aligne les textes gold et ocr avec l'algorithme choisi.

    Args:
        algo (str): Algorithme d'alignement (myers, retas, semantic).
        gold (str): Chemin vers le fichier gold aligné avec l'OCR.
        ocr (str): Chemin vers le fichier OCR aligné avec le gold.
    
    Returns:
        None.
    """
    if algo =='retas':
        RETAS(gold, ocr)
    elif algo =='myers':
        MYERS(gold, ocr)
    elif algo =='semantic':
        SEMANTIC(gold, ocr)
    else:
        print("Erreur dans l'alignement.")



def main():
    
    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Aligne un texte OCR avec son gold.')
    parser.add_argument("-a", "--algo", type=str, help="Algorithme d'alignement (myers, retas ou semantic)")
    parser.add_argument("-g", "--gold", type=str, help="Chemin vers le fichier gold")
    parser.add_argument("-o", "--ocr", type=str, help="Chemin vers le fichier OCR")
    args = parser.parse_args()

    # Vérification des arguments
    valid = check_args(args.algo, args.gold, args.ocr)
    if valid == False:
        return
    
    # Lire les textes gold et OCR
    gold_text = read_file(args.gold)
    ocr_text  = read_file(args.ocr)

    # Aligner les textes gold et OCR avec l'algorithme choisi
    align_texts(args.algo, gold_text, ocr_text)



if __name__ == "__main__":
    main()