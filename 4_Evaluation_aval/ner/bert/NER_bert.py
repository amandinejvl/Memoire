import pandas as pd
import argparse
from jiwer import wer, cer
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.metrics import precision_score, recall_score, f1_score
import spacy
nlp = spacy.load("en_core_web_sm")
from transformers import pipeline
from transformers import logging

logging.set_verbosity_error()




@dataclass
class Entity:
    sent_id: int
    gold_text: str
    gold_label: str
    ocr_text: str
    ocr_label: str



def merge_entities(entities):
    entites_fusionnees = []
    entite_courante = None

    for ent in entities:
        mot = ent["word"]
        
        if mot.startswith("##"):
            if entite_courante:
                entite_courante["word"] += mot[2:]
                entite_courante["end"] = ent["end"]
                entite_courante["score"] = max(entite_courante["score"], ent["score"])
        else:
            if entite_courante:
                entites_fusionnees.append(entite_courante)
            entite_courante = ent.copy()

    if entite_courante:
        entites_fusionnees.append(entite_courante)

    return entites_fusionnees



def extraire_entites_completes(texte, entites):
    resultat = []

    for ent in entites:
        start = ent["start"]
        end = ent["end"]

        while start > 0 and texte[start - 1].isalpha():
            start -= 1
        while end < len(texte) and texte[end].isalpha():
            end += 1

        mot_complet = texte[start:end]

        nouvelle_entite = ent.copy()
        nouvelle_entite["word"] = mot_complet
        resultat.append(nouvelle_entite)

    return resultat




def extract_entities_from_sentences(sentences):
    entities = []
    n = 0
    for sent in sentences:
        ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
        res = ner(sent)
        merged_ents = merge_entities(res)
        entities.append((n, sent, merged_ents))
        n += 1

    final_entities = []
    for ent in entities:
        entites_completes = extraire_entites_completes(ent[1], ent[2])
        final_entities.append((ent[0], ent[1], entites_completes))

    '''for avant, apres in zip(entities, final_entities):
        print(f"Avant : {avant}")
        print(f"Après : {apres}")
        print("_"*80)'''

    return final_entities


def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description="Applique une reconnaissance d'entités nommées.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv contenant les phrases gold, ocr, pré-corrigées, et corrigées")
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

    # Charger le fichier de test avec les corrections
    test = pd.read_csv(args.file)
    print(test)

    # Extraire les phrases
    gold_sentences = test['Ground Truth']
    ocr_sentences = test[colonne_name]

    # Extraire les entités nommées pour chaque phrase
    gold_ents = extract_entities_from_sentences(gold_sentences)
    ocr_ents = extract_entities_from_sentences(ocr_sentences)


    gold_sentences = []
    ocr_sentences = []

    result = []
    
    for gold_sent, ocr_sent in zip(gold_ents, ocr_ents):


        gold_entities_text = [ent['word'] for ent in gold_sent[2]]
        ocr_entities_text = [ent['word'] for ent in ocr_sent[2]]

        gold_entities = [(ent['word'], ent['entity_group']) for ent in gold_sent[2]]
        ocr_entities = [(ent['word'], ent['entity_group']) for ent in ocr_sent[2]]


        sent_id = gold_sent[0]


        print("*"*20)
        print(f"Gold : {gold_sent[0]} {gold_sent[1]}")
        print(f"OCR  : {ocr_sent[0]} {ocr_sent[1]}")
        print(f"Gold : {gold_entities}")
        print(f"OCR  : {ocr_entities}")
        print("*"*20)


        #all_entities = list(set(gold_entities + ocr_entities))

        gold_and_ocr_entities = gold_entities + ocr_entities
        all_entities = []
        for ent in gold_and_ocr_entities:
            if ent not in all_entities:
                all_entities.append(ent)
        

        final_entities = []
        associated_entities = []

        for ent in all_entities:
            if ent in gold_entities and ent in ocr_entities:
                print(f"L'entité {ent} est dans le gold et l'OCR")
                entity = Entity(sent_id, ent[0], ent[1], ent[0], ent[1])
            
            elif ent in gold_entities:

                if ent not in associated_entities:

                    print(f"L'entité {ent} est dans le gold mais pas dans l'OCR")

                    ocr_text = input("Entité correspondante dans l'OCR : ")
                    if ocr_text == "":
                        ocr_text = None
                    
                    if ocr_text != None:
                        ocr_labels = [ent for ent in all_entities if ent[0] == ocr_text]

                        if len(ocr_labels) == 1:
                            ocr_label = ocr_labels[0][1]
                        else:
                            labels = [ent[1] for ent in ocr_labels]
                            ocr_label = input(f"Choisir le label parmi {labels} : ")

                        associated_entities.append((ocr_text, ocr_label))

                    else:
                        ocr_label = None           

                    entity = Entity(sent_id, ent[0], ent[1], ocr_text, ocr_label)
            
            elif ent in ocr_entities:

                if ent not in associated_entities:

                    print(f"L'entité {ent} est dans l'OCR mais pas dans le gold")

                    gold_text = input("Entité correspondante dans le gold : ")
                    if gold_text == "":
                        gold_text = None
                    
                    if gold_text != None:
                        gold_labels = [ent for ent in all_entities if ent[0] == gold_text]

                        if len(gold_labels) == 1:
                            gold_label = gold_labels[0][1]
                        else:
                            labels = [ent[1] for ent in gold_labels]
                            gold_label = input(f"Choisir le label parmi {labels} : ")
                        
                        associated_entities.append((gold_text, gold_label))

                    else:
                        gold_label = None

                    entity = Entity(sent_id, gold_text, gold_label, ent[0], ent[1])
            
            else:
                print(f"Erreur : {ent}")
            

            if entity not in final_entities:
                final_entities.append(entity)
                #print(f"Entité {entity} ajoutée avec succès !")
                print()
        
        print("_"*80)

        result.extend(final_entities)

    df = pd.DataFrame(result, columns =['sent_id', 'gold_text', 'gold_label', 'ocr_text', 'ocr_label'])


    print(df)

    df.to_csv('output/entities.csv')



if __name__ == "__main__":
    main()



'''from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
result = ner("Apple is looking at buying U.K. startup for $1 billion")
print(result)'''


