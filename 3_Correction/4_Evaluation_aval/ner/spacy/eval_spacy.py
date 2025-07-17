import pandas as pd
import argparse
from jiwer import wer, cer
import numpy as np
import math
import pandas as pd
import Levenshtein
from dataclasses import dataclass, asdict
from sklearn.metrics import precision_score, recall_score, f1_score
import spacy
nlp = spacy.load("en_core_web_sm")



@dataclass
class Entity:
    sent_id: int
    gold_text: str
    gold_label: str
    ocr_text: str
    ocr_label: str



def df_to_objects(df, cls):
    cols = df.columns[1:]
    return [cls(**row[cols].to_dict()) for _, row in df.iterrows()]



def is_nan(x):
    return isinstance(x, float) and math.isnan(x)



def eval_simple_corres(entities):

    print("_"*50)
    print("Evaluation par simple correspondance")

    nb_vp = sum(1 for ent in entities if not pd.isna(ent.ocr_text) and not pd.isna(ent.gold_text))
    nb_fp = sum(1 for ent in entities if not pd.isna(ent.ocr_text) and pd.isna(ent.gold_text))
    nb_fn = sum(1 for ent in entities if pd.isna(ent.ocr_text) and not pd.isna(ent.gold_text))

    precision = nb_vp/(nb_vp+nb_fp)
    rappel = nb_vp/(nb_vp+nb_fn)
    f1 = 2*((precision*rappel)/(precision+rappel))

    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {rappel:.3f}")
    print(f"F-mesure  : {f1:.3f}")

    return precision, rappel, f1



def eval_corres_textes(entities):

    print("_"*50)
    print("Evaluation par correspondance des textes")

    nb_vp = sum(1 for ent in entities if ent.ocr_text == ent.gold_text)
    nb_fp = sum(1 for ent in entities if not pd.isna(ent.ocr_text) and ent.ocr_text != ent.gold_text)
    nb_fn = sum(1 for ent in entities if not pd.isna(ent.gold_text) and ent.ocr_text != ent.gold_text)    

    precision = nb_vp/(nb_vp+nb_fp)
    rappel = nb_vp/(nb_vp+nb_fn)
    f1 = 2*((precision*rappel)/(precision+rappel))

    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {rappel:.3f}")
    print(f"F-mesure  : {f1:.3f}")

    return precision, rappel, f1



###################################################################################
# All entities


def eval_corres_labels_all_entities(entities):

    print("_"*50)
    print("Evaluation par correspondance des labels")

    nb_vp = sum(1 for ent in entities if ent.ocr_label == ent.gold_label)
    nb_fp = sum(1 for ent in entities if not pd.isna(ent.ocr_label) and ent.ocr_label != ent.gold_label)
    nb_fn = sum(1 for ent in entities if not pd.isna(ent.gold_label) and ent.ocr_label != ent.gold_label)    

    precision = nb_vp/(nb_vp+nb_fp)
    rappel = nb_vp/(nb_vp+nb_fn)
    f1 = 2*((precision*rappel)/(precision+rappel))

    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {rappel:.3f}")
    print(f"F-mesure  : {f1:.3f}")

    return precision, rappel, f1



def eval_corres_textes_labels_all_entities(entities):

    print("_"*50)
    print("Evaluation par correspondance des couples (texte, label)")

    textes_labels = [((ent.gold_text, ent.gold_label), (ent.ocr_text, ent.ocr_label)) for ent in entities]

    # Remplacement des (nan, nan) par (None, None)
    cleaned_textes_labels = [
        ((None, None), second) if is_nan(first[0]) and is_nan(first[1]) else (first, second)
        for (first, second) in textes_labels
    ]

    nb_vp = sum(1 for ent in cleaned_textes_labels if ent[0] == ent[1])
    nb_fp = sum(1 for ent in cleaned_textes_labels if ent[1] != (None, None) and ent[0] != ent[1])
    nb_fn = sum(1 for ent in cleaned_textes_labels if ent[0] != (None, None) and ent[0] != ent[1])

    precision = nb_vp/(nb_vp+nb_fp)
    rappel = nb_vp/(nb_vp+nb_fn)
    f1 = 2*((precision*rappel)/(precision+rappel))

    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {rappel:.3f}")
    print(f"F-mesure  : {f1:.3f}")

    return precision, rappel, f1



def stats_corres_all_entities(entities, sentences):

    ##### Entités avec une correspondance identique #####

    print("_"*50)

    gold_ents_with_identical_corres = [ent for ent in entities if ent.gold_text == ent.ocr_text and ent.gold_label == ent.ocr_label]
    print(f"{len(gold_ents_with_identical_corres)} entités gold ont une correspondance identique dans l'OCR")

    unique_sent_ids = sorted(list(set([ent.sent_id for ent in gold_ents_with_identical_corres])))

    cers_id = []
    wers_id = []
    for s_id in unique_sent_ids:
        sent = [s for s in sentences if s[0] == s_id][0]
        cers_id.append(cer(sent[1], sent[2]))
        wers_id.append(wer(sent[1], sent[2]))

    print(f"   CER moyen gold/OCR pour les phrases contenant au moins une entité gold ayant une correspondance identique dans l'OCR : {np.mean(cers_id):.3f}")
    print(f"   WER moyen gold/OCR pour les phrases contenant au moins une entité gold ayant une correspondance identique dans l'OCR : {np.mean(wers_id):.3f}")

    ##### Entités avec une correspondance différente #####

    print("_"*50)

    gold_entities_with_different_corres = [ent for ent in entities if ent not in gold_ents_with_identical_corres and not pd.isna(ent.gold_text) and not pd.isna(ent.ocr_text)]
    print(f"{len(gold_entities_with_different_corres)} entités gold ont une correspondance différente dans l'OCR")

    distances = []
    for ent in gold_entities_with_different_corres:
        distances.append(Levenshtein.distance(ent.gold_text, ent.ocr_text))
    print(f"   Distance de Levenshtein moyenne pour les entités gold ayant une correspondance différente dans l'OCR : {np.mean(distances):.3f}")

    unique_sent_ids = sorted(list(set([ent.sent_id for ent in gold_entities_with_different_corres])))

    cers_id = []
    wers_id = []
    for s_id in unique_sent_ids:
        sent = [s for s in sentences if s[0] == s_id][0]
        cers_id.append(cer(sent[1], sent[2]))
        wers_id.append(wer(sent[1], sent[2]))

    print(f"   CER moyen gold/OCR pour les phrases contenant au moins une entité gold ayant une correspondance différente dans l'OCR : {np.mean(cers_id):.3f}")
    print(f"   WER moyen gold/OCR pour les phrases contenant au moins une entité gold ayant une correspondance différente dans l'OCR : {np.mean(wers_id):.3f}")

    ##### Entités sans correspondance dans l'OCR #####

    gold_entities_without_corres = [ent for ent in entities if not pd.isna(ent.gold_text) and pd.isna(ent.ocr_text)]
    print(f"{len(gold_entities_without_corres)} entités gold n'ont pas de correspondance dans l'OCR")

    unique_sent_ids = sorted(list(set([ent.sent_id for ent in gold_entities_without_corres])))

    cers_id = []
    wers_id = []
    for s_id in unique_sent_ids:
        sent = [s for s in sentences if s[0] == s_id][0]
        cers_id.append(cer(sent[1], sent[2]))
        wers_id.append(wer(sent[1], sent[2]))

    print(f"   CER moyen gold/OCR pour les phrases contenant au moins une entité gold sans correspondance dans l'OCR : {np.mean(cers_id):.3f}")
    print(f"   WER moyen gold/OCR pour les phrases contenant au moins une entité gold sans correspondance dans l'OCR : {np.mean(wers_id):.3f}")

    return None



def eval_all_entities(entities, sentences):

    # Evaluation par simple correspondance
    simple_corres = eval_simple_corres(entities)

    # Evaluation par correspondance des textes
    corres_textes = eval_corres_textes(entities)

    # Evaluation par correspondance des labels
    corres_labels = eval_corres_labels_all_entities(entities)

    # Evaluation par correspondance des textes et labels
    corres_textes_labels = eval_corres_textes_labels_all_entities(entities)

    # Nombre d'entités gold qui ont une correspondance identique ou différente dans l'OCR + CER et WER de la phrase
    stats_corres_all_entities(entities, sentences)

    print("_"*100)

    return None



###################################################################################
# Type entities


def eval_corres_labels_type_entities(entities, type_ent):

    print("_"*50)
    print("Evaluation par correspondance des labels")

    nb_vp = sum(1 for ent in entities if ent.ocr_label == ent.gold_label)
    nb_fp = sum(1 for ent in entities if ent.ocr_label == type_ent and ent.ocr_label != ent.gold_label)
    nb_fn = sum(1 for ent in entities if ent.gold_label == type_ent and ent.ocr_label != ent.gold_label)

    precision = nb_vp/(nb_vp+nb_fp)
    rappel = nb_vp/(nb_vp+nb_fn)
    f1 = 2*((precision*rappel)/(precision+rappel))

    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {rappel:.3f}")
    print(f"F-mesure  : {f1:.3f}")

    return precision, rappel, f1



def eval_corres_textes_labels_type_entities(entities, type_ent):

    print("_"*50)
    print("Evaluation par correspondance des couples (texte, label)")

    textes_labels = [((ent.gold_text, ent.gold_label), (ent.ocr_text, ent.ocr_label)) for ent in entities]

    # Remplacement des (nan, nan) par (None, None)
    cleaned_textes_labels = [
        ((None, None), second) if is_nan(first[0]) and is_nan(first[1]) else (first, second)
        for (first, second) in textes_labels
    ]

    nb_vp = sum(1 for ent in cleaned_textes_labels if ent[0] == ent[1])
    nb_fp = sum(1 for ent in cleaned_textes_labels if ent[1][1] == type_ent and ent[0] != ent[1])
    nb_fn = sum(1 for ent in cleaned_textes_labels if ent[0][1] == type_ent and ent[0] != ent[1])

    precision = nb_vp/(nb_vp+nb_fp)
    rappel = nb_vp/(nb_vp+nb_fn)
    f1 = 2*((precision*rappel)/(precision+rappel))

    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {rappel:.3f}")
    print(f"F-mesure  : {f1:.3f}")

    return precision, rappel, f1



def stats_corres_type_entities(entities, type_ent, sentences):

    ##### Entités avec une correspondance identique #####

    print("_"*50)

    gold_ents_with_identical_corres = [ent for ent in entities if ent.gold_label == type_ent and ent.gold_text == ent.ocr_text and ent.gold_label == ent.ocr_label]
    print(f"{len(gold_ents_with_identical_corres)} entités gold {type_ent} ont une correspondance identique dans l'OCR")

    unique_sent_ids = sorted(list(set([ent.sent_id for ent in gold_ents_with_identical_corres])))

    cers_id = []
    wers_id = []
    for s_id in unique_sent_ids:
        sent = [s for s in sentences if s[0] == s_id][0]
        cers_id.append(cer(sent[1], sent[2]))
        wers_id.append(wer(sent[1], sent[2]))

    print(f"   CER moyen gold/OCR pour les phrases contenant au moins une entité gold {type_ent} ayant une correspondance identique dans l'OCR : {np.mean(cers_id):.3f}")
    print(f"   WER moyen gold/OCR pour les phrases contenant au moins une entité gold {type_ent} ayant une correspondance identique dans l'OCR : {np.mean(wers_id):.3f}")

    ##### Entités avec une correspondance différente #####

    print("_"*50)

    gold_entities_with_different_corres = [ent for ent in entities if ent.gold_label == type_ent and ent not in gold_ents_with_identical_corres and not pd.isna(ent.ocr_text)]
    print(f"{len(gold_entities_with_different_corres)} entités gold {type_ent} ont une correspondance différente dans l'OCR")

    distances = []
    for ent in gold_entities_with_different_corres:
        distances.append(Levenshtein.distance(ent.gold_text, ent.ocr_text))
    print(f"   Distance de Levenshtein moyenne pour les entités gold {type_ent} ayant une correspondance différente dans l'OCR : {np.mean(distances):.3f}")

    unique_sent_ids = sorted(list(set([ent.sent_id for ent in gold_entities_with_different_corres])))

    cers_id = []
    wers_id = []
    for s_id in unique_sent_ids:
        sent = [s for s in sentences if s[0] == s_id][0]
        cers_id.append(cer(sent[1], sent[2]))
        wers_id.append(wer(sent[1], sent[2]))

    print(f"   CER moyen gold/OCR pour les phrases contenant au moins une entité gold {type_ent} ayant une correspondance différente dans l'OCR : {np.mean(cers_id):.3f}")
    print(f"   WER moyen gold/OCR pour les phrases contenant au moins une entité gold {type_ent} ayant une correspondance différente dans l'OCR : {np.mean(wers_id):.3f}")

    ##### Entités sans correspondance dans l'OCR #####

    gold_entities_without_corres = [ent for ent in entities if ent.gold_label == type_ent and pd.isna(ent.ocr_text)]
    print(f"{len(gold_entities_without_corres)} entités gold {type_ent} n'ont pas de correspondance dans l'OCR")

    unique_sent_ids = sorted(list(set([ent.sent_id for ent in gold_entities_without_corres])))

    cers_id = []
    wers_id = []
    for s_id in unique_sent_ids:
        sent = [s for s in sentences if s[0] == s_id][0]
        cers_id.append(cer(sent[1], sent[2]))
        wers_id.append(wer(sent[1], sent[2]))

    print(f"   CER moyen gold/OCR pour les phrases contenant au moins une entité gold {type_ent} sans correspondance dans l'OCR : {np.mean(cers_id):.3f}")
    print(f"   WER moyen gold/OCR pour les phrases contenant au moins une entité gold {type_ent} sans correspondance dans l'OCR : {np.mean(wers_id):.3f}")

    return None



def eval_type_entities(entities, type_ent, sentences):

    # Evaluation par simple correspondance
    simple_corres = eval_simple_corres(entities)

    # Evaluation par correspondance des textes
    corres_textes = eval_corres_textes(entities)

    # Evaluation par correspondance des labels
    corres_labels = eval_corres_labels_type_entities(entities, type_ent)

    # Evaluation par correspondance des textes et labels
    corres_textes_labels = eval_corres_textes_labels_type_entities(entities, type_ent)

    # Nombre d'entités gold qui ont une correspondance identique ou différente dans l'OCR + CER et WER de la phrase
    stats_corres_type_entities(entities, type_ent, sentences)

    print("_"*100)

    return None



###################################################################################



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Evalue la reconnaissance d'entités nommées.")
    parser.add_argument("-s", "--sentences", type=str, help="Chemin vers le fichier csv contenant les phrases")
    parser.add_argument("-e", "--entities", type=str, help="Chemin vers le fichier csv contenant les entités nommées")
    parser.add_argument("-c", "--colonne", type=str, help="Colonne à comparer avec le gold ('ocr', 'pre_corr', 'old_corr', 'new_corr')")
    args = parser.parse_args()

    # Nom de la colonne à comparer avec le gold
    colonne = args.colonne
    if colonne == 'ocr':
        colonne_name = 'OCR Text'
    elif colonne == 'pre_corr':
        colonne_name = 'Lexical Correction'
    elif colonne == 'old_corr':
        colonne_name = 'Old Model Correction'
    elif colonne == 'new_corr':
        colonne_name = 'New Model Correction'
    else:
        print("Nom de colonne inconnu. Choisir parmi 'ocr', 'pre_corr', 'old_corr', 'new_corr'")
        exit()

    # Charger le fichier contenant les entités
    df_entities = pd.read_csv(args.entities)
    entities = df_to_objects(df_entities, Entity)

    # Charger le fichier contenant les phrases
    df_sentences = pd.read_csv(args.sentences)
    ids = df_sentences.iloc[:, 0].tolist()
    gold_sentences = df_sentences['Ground Truth']
    ocr_sentences = df_sentences[colonne_name]
    sentences = [(s_id, gold_sent, ocr_sent) for s_id, gold_sent, ocr_sent in zip(ids, gold_sentences, ocr_sentences)]

    # CER et WER moyens des phrases
    cers = []
    wers = []
    for sent in sentences:
        cers.append(cer(sent[1], sent[2]))
        wers.append(wer(sent[1], sent[2]))
    print(f"CER moyen entre les phrases gold et OCR : {np.mean(cers):.3f}")
    print(f"WER moyen entre les phrases gold et OCR : {np.mean(wers):.3f}")

    # Evaluation toutes entités confondues
    nb_gold_all_entities = sum(1 for ent in entities if not pd.isna(ent.gold_text))
    nb_ocr_all_entities = sum(1 for ent in entities if not pd.isna(ent.ocr_text))
    print("*"*60)
    print(f"Evaluation toutes entités confondues ({nb_gold_all_entities} entités gold / {nb_ocr_all_entities} entités OCR)")
    print("*"*60)
    eval_all_entities(entities, sentences)

    # Evaluation uniquement sur les PERSON
    type_ent = 'PERSON'
    nb_gold_person = sum(1 for ent in entities if ent.gold_label == type_ent)
    nb_ocr_person = sum(1 for ent in entities if ent.ocr_label == type_ent)
    persons = [ent for ent in entities if ent.gold_label == type_ent or ent.ocr_label == type_ent]
    print("*"*60)
    print(f"Evaluation uniquement sur les personnes ({nb_gold_person} entités gold / {nb_ocr_person} entités OCR)")
    print("*"*60)
    eval_type_entities(persons, type_ent, sentences)

    # Evaluation uniquement sur les DATE
    type_ent = 'DATE'
    nb_gold_date = sum(1 for ent in entities if ent.gold_label == type_ent)
    nb_ocr_date = sum(1 for ent in entities if ent.ocr_label == type_ent)
    persons = [ent for ent in entities if ent.gold_label == type_ent or ent.ocr_label == type_ent]
    print("*"*60)
    print(f"Evaluation uniquement sur les dates ({nb_gold_date} entités gold / {nb_ocr_date} entités OCR)")
    print("*"*60)
    eval_type_entities(persons, type_ent, sentences)



'''
VP = entités de l'OCR présentes dans le gold
FP = entités de l'OCR absentes du gold
FN = entités du gold absentes de l'OCR

P = VP/(VP+FP)
R = VP/(VP+FN)
F = 2*((P*R)/(P+R))
'''



if __name__ == "__main__":
    main()