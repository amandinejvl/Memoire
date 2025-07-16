import re
import nltk
nltk.download('punkt_tab')
import csv
import Levenshtein



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



def text_to_sentences(text: str) -> list:
    """
    Segmente un texte en phrases en utilisant nltk.

    Args:
        text (str): Texte à segmenter.
    
    Returns:
        list: Liste des phrases.
    """
    sentences = nltk.sent_tokenize(text, language='french')
    return sentences



def text_to_sent_dict(text: str) -> dict:
    """
    Transforme un texte en dictionnaire de phrases avec leur position.

    Args:
        text (str): Texte à segmenter en phrases.
    
    Returns:
        dict: Dictionnaire des phrases avec leur position.
    """
    sentences = nltk.sent_tokenize(text, language='french')
    positions = {}
    start = 0
    for sentence in sentences:
        start = text.find(sentence, start)
        end = start + len(sentence) - 1
        positions[sentence] = (start, end)
        start = end + 1
    return positions



def find_corr_sent(text: str, positions: dict) -> dict:
    """
    Retrouve les phrases dans un texte à partir de leur position.
    
    Args:
        text (str): Texte dans lequel rechercher les phrases.
        positions (dict): Dictionnaire des phrases avec leur position.
    
    Returns:
        dict: Dictionnaire des phrases avec en clé la phrase du gold et en valeut la phrase de l'OCR.
    """
    substrings = {}
    for sentence, (start, end) in positions.items():
        substrings[sentence] = text[start:end+1]
    return substrings



def compute_cer(gold_sent: str, ocr_sent: str) -> float:
    """
    Calcule le taux d'erreur par caractère (CER) entre deux phrases en utilisant la distance de Levenstein.

    Args:
        gold_sent (str): Phrase de référence.
        ocr_sent (str): Phrase ocr.
    
    Returns:
        float: Taux d'erreur par caractère.
    """
    lev_distance = Levenshtein.distance(gold_sent, ocr_sent)
    cer = lev_distance / max(len(gold_sent), 1)
    return cer



def save_to_csv(results: list, filename: str) -> None:
    """
    Sauvegarde les résultats de l'alignement dans un fichier CSV.
    
    Args:
        results (list): Liste des résultats à enregistrer, sous forme de tuples (gold_sent, ocr_sent, cer).
        filename (str): Nom du fichier CSV.
    
    Returns:
        None.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["gold_sent", "ocr_sent", "CER"])
        writer.writerows(results)
    print(f"Les résultats ont bien été enregistrés dans le fichier {filename}.")



def save_to_txt(text: str, filename: str) -> None:
    """
    Sauvegarde un texte dans un fichier texte.

    Args:
        text (str): Texte à sauvegarder.
        filename (str): Nom du fichier.
    
    Returns:
        None
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Le texte a bien été enregistré dans le fichier {filename}.")