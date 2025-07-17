from transformers import AutoTokenizer, BartTokenizer, BertTokenizer, BertForMaskedLM
import Levenshtein
import difflib
import torch
import torch.nn.functional as F
import csv
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from parallelbar import progress_map
import numpy as np
from jiwer import wer, cer
from nltk.tokenize import word_tokenize
import re
from dataclasses import dataclass
from collections import Counter
from typing import List
from collections import defaultdict



from transformers.utils import logging
logging.set_verbosity_error()



@dataclass(frozen=True)
class candidate:
    word: str
    bert_score: float
    distance: float
    lcs: float
    final: float



def lcs(a: str, b: str) -> str:
    m, n = len(a), len(b)
    # Création d'une matrice (m+1) x (n+1) remplie de 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Remplissage de la matrice
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    # Reconstruction de la sous-séquence commune
    i, j = m, n
    lcs_seq = []
    while i > 0 and j > 0:
        if a[i-1] == b[j-1]:
            lcs_seq.append(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs_seq))



def get_vocabulary(file: str) -> list:
    """Charge le vocabulaire à partir d'un fichier csv"""
    existing_words = []
    with open(file, mode ='r') as f:
        words = csv.reader(f)
        for word in words:
            existing_words.append(word[0].lower())
    return existing_words



def rebuild_words(tokens: list) -> dict:
    """Reforme les mots à partir des tokens"""
    words = []
    for tok in tokens:
        if not tok.startswith("▁") and words: 
            if tok.replace("▁", "").isalnum() or tok == "-" or tok =="’" or tok == "'":
                words[-1] = words[-1] + "|" + tok
            else:
                words.append(tok)
        else:
            words.append(tok)
            
    result = []
    for word in words:
        result.append((word.replace("|", ""), word.split("|")))
    return result



def get_closest_words(target_word, word_list, k=50):
    """Trouve les k mots ayant la distance d'édition la plus petite avec un mot"""
    distances = [(word, Levenshtein.distance(target_word, word)) for word in word_list]
    distances.sort(key=lambda x: x[1])
    words = [cand[0] for cand in distances]
    return words[:k]





def get_most_likely_words(sentence: str, word: str, k=50, weight=10) -> list:
    """Trouve les mots les plus probables pour remplacer un mot dans une phrase"""

    # Charger BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    # Masquer le mot dans la phrase
    word_to_replace = word.replace("▁", "")
    pattern = rf'\b{re.escape(word_to_replace)}\b'
    sentence = re.sub(pattern, "[MASK]", sentence)

    # Tokeniser
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Extraire les logits pour le token masqué
    mask_logits = logits[0, mask_token_index, :].squeeze()

    # Obtenir les scores de probabilité
    probs = F.softmax(mask_logits, dim=-1)

    # Top k mots les plus probables
    top_k = k
    top_probs, top_indices = torch.topk(probs, top_k)

    # Afjouter les tokens et leur score à la liste des candidats
    top_k_words = []
    for i in range(top_k):
        token = tokenizer.decode([top_indices[i]])
        if "#" not in token:
            score = top_probs[i].item()
            #top_k_words.append((token, score*weight))
            top_k_words.append(
                candidate(
                    word=token, 
                    bert_score=score*weight, 
                    distance=None,
                    lcs=None, 
                    final=None
                )
            )

    return top_k_words



def get_bert_prob(sentence: str, orig_word: str, target_word: str, weight=10):
    """Donne le score de probabilité d'un mot à la place du mot d'origine"""

    # Charger BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    # Masquer le mot dans la phrase
    word_to_replace = orig_word.replace("▁", "")
    pattern = rf'\b{re.escape(word_to_replace)}\b'
    sentence = re.sub(pattern, "[MASK]", sentence)

    # Tokeniser
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Extraire les logits pour le token masqué
    mask_logits = logits[0, mask_index, :].squeeze(0)

    # Obtenir le score de probabilité
    probs = F.softmax(mask_logits, dim=-1)

    # ID du mot cible
    token_id = tokenizer.convert_tokens_to_ids(target_word)

    # Probabilité du mot à la position masquée
    word_prob = probs[token_id].item()

    return word_prob*weight



def get_lcs(orig_word: str, candidate: str, weight=0.1) -> float:
    """Retourne l'inverse de la longueur de la plus longue sous-séquence commune entre deux mots"""
    seq = difflib.SequenceMatcher(None, orig_word, candidate)
    match = seq.find_longest_match(0, len(orig_word), 0, len(candidate))
    lcs_length = match.size
    return lcs_length*weight



def get_lev_distance(orig_word: str, candidate: str, weight=0.1) -> float:
    """Retourne l'inverse de la distance de Levenstein entre deux mots"""
    distance = -Levenshtein.distance(orig_word, candidate) #/len(orig_word)
    return distance*weight



def add_likely_words(orig_word, likely_words):
    """Ajoute les mots probables à la liste des candidats"""

    liste = []

    # Parcourir tous les mots probables
    for i in range(len(likely_words)):

        # Si la différence de longueur entre le candidat et le mot d'origine est inférieure ou égale à 70% de la longueur du mot d'origine
        if abs(len(orig_word)-len(likely_words[i].word)) <= round(len(orig_word)-0.7*len(orig_word)):

            # Calculer la longueur de la plus longue sous-séquence commune entre le mot d'origine et le candidat
            lcs = get_lcs(orig_word, likely_words[i].word)

            if lcs > 0:

                # Extraire le score de probabilité du mot d'après BERT
                bert_score = likely_words[i].bert_score

                # Calculer sa distance d'édition avec le mot d'origine
                distance = get_lev_distance(orig_word, likely_words[i].word)

                # Ajouter le mot à la liste des candidats
                liste.append(
                    candidate(
                        word=likely_words[i].word, 
                        bert_score=bert_score, 
                        distance=distance, 
                        lcs=lcs, 
                        final=None,
                    )
                )
    
    return liste



def add_closest_words(sentence, orig_word, closest_words):
    """Ajoute les mots proches à la liste des candidats"""

    liste = []

    # Parcourir tous les mots proches
    for i in range(len(closest_words)):

        # Si la différence de longueur entre le candidat et le mot d'origine est inférieure ou égale à 70% de la longueur du mot d'origine
        if abs(len(orig_word)-len(closest_words[i])) <= 0.7*len(orig_word):

            # Calculer la longueur de la plus longue sous-séquence commune entre le mot d'origine et le candidat
            lcs = get_lcs(orig_word, closest_words[i])

            if lcs > 0:

                # Calculer son score de probabilité d'après BERT
                bert_score = get_bert_prob(sentence, orig_word, closest_words[i])

                # Calculer sa distance d'édition avec le mot d'origine
                distance = get_lev_distance(orig_word, closest_words[i])

                # Ajouter le mot à la liste des candidats
                liste.append(
                    candidate(
                        word=closest_words[i], 
                        bert_score=bert_score, 
                        distance=distance, 
                        lcs=lcs, 
                        final=None
                    )
                )

    return liste



def find_dupplicated_candidates(liste):
    """Supprime les doublons en multipliant leur score bert par 10"""
    
    # Compter les occurrences des mots
    word_counts = Counter(c.word for c in liste)

    # Pour stocker les résultats finaux
    result = []

    # Pour suivre les mots déjà traités (afin de garder un seul exemplaire)
    seen_words = set()

    for c in liste:
        if c.word not in seen_words:
            if word_counts[c.word] > 1:
                # Multiplier bert_score par 10 pour ce candidat
                new_c = candidate(
                    word=c.word,
                    bert_score=c.bert_score * 10,
                    distance=c.distance,
                    lcs=c.lcs,
                    final=c.final,
                )
                result.append(new_c)
            else:
                # Cas où le mot n'apparaît qu'une fois, on garde tel quel
                result.append(c)
            seen_words.add(c.word)
    
    return result



def filter_candidates(candidates):
    """Filtre et calcule le score final des candidats"""

    # Initialiser une liste pour stocker les candidats finaux avec leur score final
    final_candidates = []

    # Parcourir tous les candidats
    for cand in candidates:

        word=cand.word
        bert_score=cand.bert_score
        distance=cand.distance
        lcs=cand.lcs

        # Si le score Lev est supérieur à -0.5
        if distance > -0.5:
    
            # Additionner les 3 scores
            final_score = bert_score + distance + lcs
    
            # Ajouter le mot et ses scores ains que le score final à la liste
            final_candidates.append(
                candidate(
                    word=word, 
                    bert_score=bert_score, 
                    distance=distance, 
                    lcs=lcs, 
                    final=final_score
                )
            )
    
    return final_candidates



def find_most_freq_first(candidates):
    """Trouve le ou les candidats qui ont eu les meilleurs scores le plus de fois"""

    # Dictionnaire pour compter les premières places
    first_place_counts = defaultdict(int)

    # Liste des scores
    scores = ['bert_score', 'distance', 'lcs', 'final']

    # Pour chaque score, trouver la meilleure note et qui l'a obtenue
    for score in scores:
        max_score = max(getattr(c, score) for c in candidates)
        for cand in candidates:
            if getattr(cand, score) == max_score:
                first_place_counts[cand] += 1

    # Trouver le nombre maximal de premières places
    max_firsts = max(first_place_counts.values())

    # Lister tous les candidats ayant ce score
    best_candidates = [cand for cand, count in first_place_counts.items() if count == max_firsts]

    return best_candidates



def find_best_candidate(sentence, orig_word, three_best_candidates):
    """Donne la meilleure correction entre le mot d'origine et les 3 meilleurs candidats"""

    # Donner un score BERT au mot d'origine
    bert_score = get_bert_prob(sentence, orig_word, orig_word)


    # Si la liste des candidats n'est pas vide
    if three_best_candidates != []:

        # Extraire les scores bert des candidats
        bert_scores = [c.bert_score for c in three_best_candidates]

        # Si l'écart-type des scores bert est inférieur à 2
        if np.std(bert_scores) < 2:

            # Trouver le ou les candidats qui ont eu les meilleurs scores le plus de fois
            premiers = find_most_freq_first(three_best_candidates)

            # Si le score BERT du mot d'origine est supérieur à celui du premier
            if bert_score > premiers[0].bert_score:

                # Si le score Lev du meilleur candidat est inférieur ou égal à 0.1, on garde le meilleur candidat
                if premiers[0].distance <= 0.1:
                    corr = premiers[0].word

                # Sinon, on garde le mot d'origine
                else:
                    corr = orig_word

            # Sinon, on garde le meilleur candidat
            else:
                corr = premiers[0].word
            
        # Sinon (si l'écart-type des scores bert est supérieur ou égal à 2)
        else:
            
            # On garde le candidat au score final le plus élevé
            corr = three_best_candidates[0].word

    # Sinon (si la liste des candidats est vide), on garde le mot d'origine
    else:
        corr = orig_word
    
    return corr.replace("_", "")



def compare_lexique_context(sentence: str, orig_word: str, closest_words: list, likely_words: list):
    """Trouve le mot le plus probable en comparant les mots proches et les mots probables"""

    orig_word = orig_word.replace("▁", "")

    print(f"Mot d'origine : {orig_word}")

    # Ajouter les mots probables à la liste des candidats
    scores_likely_words = add_likely_words(orig_word, likely_words)

    # Ajouter les mots proches à la liste des candidats
    scores_closest_words = add_closest_words(sentence, orig_word, closest_words)

    # Fusionner les deux listes de candidats
    all_candidates = scores_likely_words + scores_closest_words

    # Trouver les mots qui apparaissent plusieurs fois et multiplier leur score bert par 10
    unique_candidates = find_dupplicated_candidates(all_candidates)

    # Filtrer et calculer le score final des candidats
    final_candidates = filter_candidates(unique_candidates)

    # Trier les candidats par score final croissant
    sorted_candidates = sorted(final_candidates, key=lambda x: x.final, reverse=True)

    # Extraire les 3 candidats ayant le score final le plus élevé
    three_best_candidates = sorted_candidates[0:3]

    # Trouver le meilleur candidat entre les meilleurs candidats et le mot d'origine
    corr = find_best_candidate(sentence, orig_word, three_best_candidates)

    print(f"Correction : {corr}")

    # Retourner le meilleur candidat
    return corr



def special_char(toks: list) -> bool:
    """Détermine si on doit intervenir sur le mot"""

    # Partir du principe que l'on doit intervenir
    do_something = True

    # Supprimer les marqueurs de début de mot
    simple_toks = [tok.replace("▁", "") for tok in toks]
    word = ''.join(simple_toks)
    chiffres = [str(i) for i in range(0, 10)]

    # Si le mot valide l'une des conditions suivantes, on n'intervient pas
    if word != "":
        if ":" in word:
            do_something = False
        if "-" in word:
            do_something = False
        if "—" in word:
            do_something = False
        if "’" in word:
            do_something = False
        if "`" in word:
                do_something = False
        if "'" in word:
            do_something = False
        if "“" in word:
            do_something = False
        if "‘" in word:
            do_something = False
        if "<" in word:
            do_something = False
        if ">" in word:
            do_something = False
        if word[0].isupper():
            do_something = False
        if set(simple_toks) & set(chiffres):
            do_something = False
    else:
        do_something = False
        
    return do_something



def count_matches(text: str, word: str) -> int:
    """Compte le nombre d'apparition d'un mot dans un texte"""
    word_to_replace = word.replace("▁", "")
    pattern = rf'\b{re.escape(word_to_replace)}\b'
    matches = re.findall(pattern, text)
    return len(matches)



def correct_sentence(sent: str, ref_words:list, tokenizer) -> str:
    """Corrige une phrase"""

    # Tokeniser la phrase et reconstituer les mots
    tokens = tokenizer.tokenize(sent)
    words = rebuild_words(tokens)
    #words = word_tokenize(sent)

    # Initialiser une liste pour stocker les mots de la phrase corrigée
    corrected_sentence = []

    # Parcourir tous les mots de la phrase
    for word, toks in words:

        # Si le mot n'existe pas dans le lexique
        if word.replace("▁", "").lower() not in ref_words:

            # Si le mot est constitué d'un seul token
            if len(toks) == 1:

                corrected_sentence.append(word)
            
            # Sinon (si le mot contient plusieurs tokens)
            else:

                # Compter le nombre de fois qu'apparaît le mot dans la phrase
                nb_occ = count_matches(sent, word)

                # Si le mot n'apparaît qu'une fois
                if nb_occ == 1 and special_char(toks) == True:

                    # Trouver les mots existants les plus proches
                    closest_words = get_closest_words(word, ref_words)

                    # Si le mot n'a pas de correction identique
                    if word.lower().replace("▁", "") not in closest_words:

                        # Trouver les mots les plus probables pour le remplacer
                        likely_words = get_most_likely_words(sent, word)

                        # Comparer les mots les plus proches avec les plus probables
                        correction = compare_lexique_context(sent, word, closest_words, likely_words)

                        # Rajouter le tiret de début de mot au mot corrigé si nécessaire
                        if word.startswith("▁"):
                            if word.replace("▁", "")[0].isupper():
                                correct_word = "▁" + correction.capitalize()
                            else:
                                correct_word = "▁" + correction
                        else:
                            if word[0].isupper():
                                correct_word = correction.capitalize()
                            else:
                                correct_word = correction
                        
                        # Ajouter le mot corrigé à la suite
                        corrected_sentence.append(correct_word)
                    
                    # Si le mot a une correction identique
                    else:
                        
                        # Ajouter le mot à la suite
                        corrected_sentence.append(word)

                # Sinon (si le mot apparaît plus qu'une fois dans la phrase)
                else:

                    # Ajouter le mot à la suite
                    corrected_sentence.append(word)
            
        # Sinon (si le mot appartient au lexique)
        else:
            
            # Ajouter le mot à la suite
            corrected_sentence.append(word)
    
    # Reformer la phrase
    final_sentence = ''.join(corrected_sentence).replace("▁", " ")[1::]

    if sent != final_sentence:
        print(f"Phrase OCR : {sent}")
        print(f"Correction : {final_sentence}")

    return final_sentence



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Applique la correction avec un modèle et calcule l'amélioration.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv sur lequel appliquer la correction")
    parser.add_argument("-l", "--lexique", type=str, help="Chemin vers le fichier csv contenant le lexique de référence")

    args = parser.parse_args()

    # Charger le fichier de test
    test = pd.read_csv(args.file)
    #test = test.iloc[[67, 12, 16, 17, 25, 9, 31]]

    # Extraire les phrases OCR
    ocr_sentences = test['OCR Text']

    # Extraire les phrases gold
    gold_sentences = test['Ground Truth']

    # Charger le tokeniser du modèle
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    # Charger le vocabulaire du tokenizer
    vocab = tokenizer.get_vocab()

    # Charger le lexique de référence
    lexique_file = args.lexique
    lexique = get_vocabulary(lexique_file)

    # Initialiser une liste pour stocker les phrases corrigées
    corrected_sentences = []

    # Nombre de phrases modifiées
    nb_modifs = 0
    old_wers = []
    new_wers = []

    # Corriger les phrases
    for gold, sent in tqdm(zip(gold_sentences, ocr_sentences)):
        print(sent)
        corr_sent = correct_sentence(sent, lexique, tokenizer)
        corrected_sentences.append(corr_sent)
        
        if corr_sent != sent:
            nb_modifs += 1
            old_wer = wer(gold, sent)
            new_wer = wer(gold, corr_sent)
            old_wers.append(old_wer)
            new_wers.append(new_wer)
        
        print("_"*80)

    print()
    print(f"{nb_modifs} phrases modifiées.")
    print(f"Moyenne des WER gold/OCR pour les phrases modifiées : {np.mean(old_wers)}")
    print(f"Moyenne des WER gold/correction pour les phrases modifiées : {np.mean(new_wers)}")
    print()

    # Ajouter la colonne contenant les corrections au Dataframe
    test['Lexical Correction'] = corrected_sentences

    # Sauvegarder la correction dans un fichier csv
    test.to_csv('lexical_correction.csv')
    print("Les corrections ont bien été enregistrées dans le fichier lexical_correction.csv.")



if __name__ == "__main__":
    main()
