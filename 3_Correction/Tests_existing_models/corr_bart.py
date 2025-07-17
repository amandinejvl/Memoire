from datasets import Dataset
from IPython.core.getipython import get_ipython
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, pipeline
import Levenshtein
import pandas as pd
import torch
import argparse



def cer(prediction: str, target: str) -> float:
    """
    Calcule le taux d'erreur par caractère entre la prédiction et la cible.

    Args:
        prediction (str): Texte correspondant à la prédiction (de l'OCR ou du modèle de correction).
        target (str): Texte cible (gold).

    Returns:
        float: CER entre la prédiction et la cible.
    """
    distance = Levenshtein.distance(prediction, target)
    return distance / len(target)



def get_results(data: Dataset, preds: list) -> pd.DataFrame:
    """
    Sauvegarde les résultats de la correction et calcule la réduction des CER.

    Args:
        data (Dataset): Dataset sur lequel a été appliquée la correction.
        preds (list): Liste des prédictions de correction du modèle.

    Returns:
        pd.DataFrame: DataFrame de résultats de la correction.
    """
    results = data.to_pandas()
    results['Model Correction'] = preds
    results['old_CER'] = results.apply(lambda row: cer(row['OCR Text'], row['Ground Truth']), axis=1)
    results['new_CER'] = results.apply(lambda row: cer(row['Model Correction'], row['Ground Truth']), axis=1)
    results['CER_reduction'] = ((results['old_CER'] - results['new_CER']) / results['old_CER']) * 100
    return results



def check_args(file: str, model: str) -> bool:
    """
    Vérifie les arguments fournis par l'utilisateur.

    Args:
        file (str): Chemin vers le fichier csv sur lequel appliquer la correction.
        model (str): Nom du modèle.

    Returns:
        bool: True si les arguments sont valides, False sinon.
    """

    # Partir du principe que les arguments sont valides
    valid_args = True

    # Vérifier gold et ocr
    if not file:
        print("Veuillez fournir le chemin vers le fichiers sur lequel vous souhaitez appliquer la correction.")
        valid_args = False

    # Verifier l'algorithme d'alignement choisi
    if model not in ['bart-base','bart-large']:
        print("Modèle inconnu. Choisissez bart-base ou bart-large.")
        valid_args = False

    return valid_args



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Applique la correction avec un modèle et calcule l'amélioration.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv sur lequel appliquer la correction")
    parser.add_argument("-m", "--model", type=str, help="Nom du modèle ('bart-base' ou 'bart-large')")
    args = parser.parse_args()

    # Vérification des arguments
    valid = check_args(args.file, args.model)
    if valid == False:
        return    
    
    # Chargement du fichier de test
    test = pd.read_csv(args.file)
    test = Dataset.from_pandas(test)

    # Chargement du modèle
    model_name = args.model
    model_dir = f'pykale/{model_name}-ocr'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    generator = pipeline('text2text-generation', model=model.to('cuda'), tokenizer=tokenizer, device='cuda', max_length=1024)

    # Application de la correction au fichier
    preds = []
    for sample in tqdm(test):
        preds.append(generator(sample['OCR Text'])[0]['generated_text'])

    # Sauvegarde des résultats
    results = get_results(test, preds)
    results.to_csv(f'output/result_{model_name}.csv', index=False)

    # Affichage des 10 premières lignes du fichier de résultats
    corrections = pd.read_csv(f'output/result_{model_name}.csv')
    print(corrections.head(10))



if __name__ == "__main__":
    main()
