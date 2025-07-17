import pandas as pd
import argparse
from jiwer import wer, cer
import numpy as np
import pandas as pd
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



def extract_entities_from_sentences(sentences):
    entities = []
    n = 0
    for sent in sentences:
        doc = nlp(sent)
        ents = []
        for ent in doc.ents:
            ents.append(ent)
        entities.append((n, sent, ents))
        n += 1
    return entities



def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description="Applique une reconnaissance d'entités nommées.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv contenant les phrases gold, ocr, pré-corrigées, et corrigées")
    args = parser.parse_args()

    # Charger le fichier de test avec les corrections
    test = pd.read_csv(args.file)
    print(test)

    # Extraire les phrases
    gold_sentences = test['Ground Truth']
    ocr_sentences = test['OCR Text']
    pre_corr_sentences = test['Lexical Correction']
    old_corr_sentences = test['Old Model Correction']
    new_corr_sentences = test['New Model Correction']

    # Extraire les entités nommées pour chaque phrase
    gold_ents = extract_entities_from_sentences(gold_sentences)
    ocr_ents = extract_entities_from_sentences(ocr_sentences)
    pre_corr_ents = extract_entities_from_sentences(pre_corr_sentences)
    old_corr_ents = extract_entities_from_sentences(old_corr_sentences)
    new_corr_ents = extract_entities_from_sentences(new_corr_sentences)


    gold_sentences = []
    ocr_sentences = []

    result = []



    
    for gold_sent, ocr_sent in zip(gold_ents, new_corr_ents):

        gold_entities_text = [ent.text for ent in gold_sent[2]]
        ocr_entities_text = [ent.text for ent in ocr_sent[2]]

        gold_entities = [(ent.text, ent.label_) for ent in gold_sent[2]]
        ocr_entities = [(ent.text, ent.label_) for ent in ocr_sent[2]]


        sent_id = gold_sent[0]


        print("*"*20)
        print(f"Gold : {gold_sent[0]} {gold_sent[1]}")
        print(f"OCR  : {ocr_sent[0]} {ocr_sent[1]}")
        print(f"Gold : {gold_entities_text}")
        print(f"OCR  : {ocr_entities_text}")
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