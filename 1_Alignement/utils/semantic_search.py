import sys
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import nltk
import re
import csv
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import argparse



try:
    # import pour l'appel depuis alignment.py
    from .process_texts import *  
except ImportError:
    # import pour l'appel direct de RETAS.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from process_texts import *



# Charger BERT et le tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)



def get_bert_embeddings(sentences: list) -> np.ndarray:
    """
    Retourne les embeddings des phrases en utilisant le token [CLS]

    Args:
        sentences (list): Liste des phrases.
    
    Returns:
        np.ndarray: Embeddings des phrases.
    """
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Prendre l'embedding du token [CLS] (premier token)
    embeddings = outputs.last_hidden_state[:, 0, :]
    
    return embeddings.numpy()



def visualiser_matrice(matrix: np.ndarray) -> None:
    """
    Visualise une matrice de similarité.
    
    Args:
        matrix (np.ndarray): Matrice de similarité.
    
    Returns:
        None.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='Reds', annot=False)
    plt.title("Matrice de similarité")
    plt.savefig("output/matrice.png")



def SEMANTIC(gold_text: str, ocr_text: str) -> None:
    """
    Aligne un texte OCR avec son gold en utilisant la Semantic Search.

    Args:
        gold_text (str): Texte gold.
        ocr_text (str): Texte OCR.
    
    Returns:
        None.
    """

    # Segmenter les textes en phrases
    gold_sentences = text_to_sentences(gold_text)
    ocr_sentences = text_to_sentences(ocr_text)
    print("Nombre de phrases gold :", len(gold_sentences))
    print("Nombre de phrases ocr  :", len(ocr_sentences))

    # Calculer les embeddings avec BERT
    embeddings_gold = get_bert_embeddings(gold_sentences)
    embeddings_ocr = get_bert_embeddings(ocr_sentences)

    # Calculer la similarité cosinus
    similarity_matrix = cosine_similarity(embeddings_gold, embeddings_ocr)
    visualiser_matrice(similarity_matrix)

    # Trouver l'index de la phrase la plus similaire pour chaque phrase de gold_sentences
    best_matches = np.argmax(similarity_matrix, axis=1)

    # Créer une liste de paires (gold_sent, ocr_sent) avec les meilleures correspondances
    paires = []
    for i, j in enumerate(best_matches):
        gold_sent = gold_sentences[i]
        ocr_sent = ocr_sentences[j]
        paires.append((gold_sent, ocr_sent))
    
    # Préparer les résultats pour sauvegarder dans le CSV
    results = []
    for paire in paires:
        cer_score = compute_cer(paire[0], paire[1])

        # Ajouter les résultats à la liste
        results.append([paire[0], paire[1], cer_score])

    # Sauvegarder les résultats dans un fichier CSV
    save_to_csv(results, 'output/results_semantic_search.csv')



if __name__ == "__main__":

    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Aligne un texte OCR avec son gold en utilisant la semantic search de Debaene.')
    parser.add_argument("-g", "--gold", type=str, help="Chemin vers le fichier gold")
    parser.add_argument("-o", "--ocr", type=str, help="Chemin vers le fichier OCR")
    args = parser.parse_args()

    # Vérification des arguments
    if not args.gold or not args.ocr:
        print("Veuillez fournir les chemins vers les fichiers gold et OCR.")
        exit()

    # Lire les fichiers gold et OCR
    gold_text = read_file(args.gold)
    ocr_text = read_file(args.ocr)

    SEMANTIC(gold_text, ocr_text)