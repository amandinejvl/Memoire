import csv
import pandas as pd
from datasets import Dataset
import argparse
import numpy as np
from jiwer import wer, cer
import difflib



def get_old_words(sent1: str, sent2: str) -> list:
    """Trouve les mots supprimés de la phrase 1 à 2"""
    words1 = sent1.split()
    words2 = sent2.split()
    diff = difflib.ndiff(words1, words2)
    old = []
    for word in diff:
        if word.startswith("- "):
            old.append(word[2:])
    return old



def get_new_words(sent1: str, sent2: str) -> list:
    """Trouve les mots ajoutés de la phrase 1 à 2"""
    words1 = sent1.split()
    words2 = sent2.split()
    diff = difflib.ndiff(words1, words2)
    new = []
    for word in diff:
        if word.startswith("+ "):
            new.append(word[2:])
    return new



def compare_sentences(sent1: str, sent2: str) -> tuple:
    """Compare les mots ajoutés et supprimés entre les 2 phrases"""
    old_words = get_old_words(sent1, sent2)
    new_words = get_new_words(sent1, sent2)
    return (len(new_words), old_words, new_words)



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Analyse les résultats de la correction.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv contenant la correction")
    args = parser.parse_args()

    # Charger le fichier contenant la correction
    df = pd.read_csv(args.file)

    # Extraire les id
    ids = df[df.columns[0]].tolist()

    # Extraire les phrases OCR
    ocr_sentences = df['OCR Text']

    # Extraire les phrases gold
    gold_sentences = df['Ground Truth']

    # Extraire les phrases corrigées
    corr_sentences = df['Lexical Correction']

    # Nombre de phrases modifiées
    nb_modifs = 0
    modified_words = 0
    words_to_modify = 0
    right_mod = 0
    improvements = 0
    degradations = 0
    no_change = 0

    old_wers = []
    new_wers = []

    # Parcourir toutes les phrases
    for id, gold, ocr, corr in zip(ids, gold_sentences, ocr_sentences, corr_sentences):
        
        # Si la phrase a été modifiée
        if ocr != corr:
            nb_modifs += 1

            # Calculer l'ancien et le nouveau WER
            old_wer = wer(gold, ocr)
            new_wer = wer(gold, corr)
            old_wers.append(old_wer)
            new_wers.append(new_wer)

            # Trouver les mots modifiés
            nb_modif_words, ocr_mod_words, corr_words = compare_sentences(ocr, corr)
            modified_words += nb_modif_words

            nb_wrong_gold_words, gold_words, ocr_words = compare_sentences(gold, ocr)
            words_to_modify += nb_wrong_gold_words

            words_to_correct = [(gold, ocr) for gold, ocr in zip(gold_words, ocr_words)]
            corrected_words = [(ocr, corr) for ocr, corr in zip(ocr_mod_words, corr_words)]

            # Trouver les améliorations/dégradations/absences de correction
            for gold_ocr in words_to_correct:
                if gold_ocr[1] in ocr_mod_words:
                    right_mod += 1
                    if gold_ocr[0] in corr_words:
                        improvements += 1
                    else:
                        degradations += 1
                else:
                    no_change += 1

            # Afficher les modifications
            for ocr, corr in zip(ocr_mod_words, corr_words):
                print(f"{id} : {ocr} --> {corr}")



            print("_"*80)

            


        
    print()
    print(f"{nb_modifs} phrases modifiées.")
    print(f"Moyenne des WER gold/OCR pour les phrases modifiées : {np.mean(old_wers)}")
    print(f"Moyenne des WER gold/correction pour les phrases modifiées : {np.mean(new_wers)}")
    print()

    print()
    print(f"{modified_words} mots ayant subi une correction")
    print()

    print(f"{words_to_modify} mots à corriger")



    print(f"├── {improvements} mots bien corrigés")
    print(f"├── {degradations} mots mal corrigés")
    print(f"└── {no_change} mots non corrigés")
    print()
    print(f"{modified_words-right_mod} mots modifiés à tort")

    print()



if __name__ == "__main__":
    main()
