from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy
from lemminflect import getAllInflections
import csv
from collections import Counter
from tqdm import tqdm
import argparse



# Charger le modèle spaCy anglais
nlp = spacy.load("en_core_web_sm")



def get_wordnet_lemmas():
    """Extrait tous les lemmes de wordnet"""
    words = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            words.add(lemma.name())
    return list(words)



def get_inflections(word: str) -> list:
    """Renvoie les formes fléchies d'un mot donné"""
    doc = nlp(word)
    token = doc[0]
    pos = token.pos_

    all_forms = getAllInflections(word)

    if all_forms != {}:
        forms = set()
        for form_list in all_forms.values():
            for form in form_list:
                forms.add(form)
    else:
        forms = [word]
   
    return sorted(forms)



def read_file(filename: str) -> str:
    """Lit le contenu d'un fichier et remplace les retours à la ligne par un espace sauf quand il est précédé d'un espace (word tronqué par souci de mise en page)"""
    try:
        with open(filename, 'r') as file:
            content = file.read()
            content_without_retours = content.replace("-\n", "").replace("\n", " ")
            #final_content = re.sub(r"(?=[.,;:!?])", " ", content_without_retours)
            return content_without_retours
    except FileNotFoundError:
        print(f"The file '{filename}' was not found.")
        return None



def get_frequence_words(texte: str, k=200) -> list: 
    """Renvoie les k mots les plus fréquents d'un texte"""

    # Tokeniser le texte
    words = word_tokenize(texte)

    # Supprimer les stopwords
    stop_words = set(stopwords.words("english"))  # ou "french" selon la langue
    words_utils = [word for word in words if word.isalpha() and word not in stop_words]

    # Compter les fréquences des mots et les trier de façon décroissante
    compteur = Counter(words_utils)
    sorted_dico = dict(sorted(compteur.items(), key=lambda item: item[1], reverse=True))

    # Mettre les mots en minuscules
    words = [word.lower() for word in sorted_dico.keys()]

    return words



def get_characters(texte: str) -> list:
    """Extrait les personnages d'un texte"""
    doc = nlp(texte)
    characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    unique_characters = []
    for char in characters:
        if char not in unique_characters:
            unique_characters.append(char)
    return unique_characters



def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description="Crée un lexique personnalisé à partir de la transcription OCR.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier txt contenant la transcription OCR")
    args = parser.parse_args()

    # Charger le vocabulaire de wordnet
    wordnet_words = get_wordnet_lemmas()

    # Trouver toutes les formes de chaque mot de wordnet et les ajouter au lexique
    lexique = []
    for word in tqdm(wordnet_words):
        inflections = get_inflections(word)
        lexique.extend(inflections)

    # Charger le texte issu de la transcription OCR
    texte = read_file(args.file)

    # Extraire les personnages et les ajouter au lexique
    characters = get_characters(texte)
    lexique.extend(characters)

    # Ajouter les 10 mots les plus fréquents de la transcription OCR que ne sont pas déjà dans le lexique
    n = 0
    most_common_words = get_frequence_words(texte)
    for word in most_common_words:
        if n <= 10:
            if word not in lexique:
                print(word)
                lexique.append(word)
                n += 1
        else:
            break
    
    # Sauvegarder le lexique dans un fichier csv
    dest_file = 'lexique.csv'
    with open(dest_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for word in lexique:
            writer.writerow([word])
        print(f"Le lexique a bien été enregistré dans le fichier {dest_file}.")



if __name__ == "__main__":
    main()