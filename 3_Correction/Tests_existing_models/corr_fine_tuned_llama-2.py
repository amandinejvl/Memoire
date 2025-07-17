from datasets import Dataset
from IPython.core.getipython import get_ipython
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, pipeline, AutoModelForCausalLM
import Levenshtein
import pandas as pd
import torch
import argparse
from huggingface_hub import login
login(token = 'your_token')



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
    models = [
        'llama-2-7b-ocr-5-epochs-tesseract', 
        'llama-2-7b-ocr-5-epochs-kraken', 
        'llama-2-7b-ocr-10-epochs-tesseract', 
        'llama-2-7b-ocr-10-epochs-kraken', 
        'llama-2-13b-ocr-5-epochs-tesseract', 
        'llama-2-13b-ocr-5-epochs-kraken', 
        'llama-2-13b-ocr-10-epochs-tesseract', 
        'llama-2-13b-ocr-10-epochs-kraken'
    ]
    
    if model not in models:
        print(f"Modèle inconnu. Choisissez parmi :")
        print("\n".join(models))
        valid_args = False

    return valid_args



def main():

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Applique la correction avec un modèle et calcule l'amélioration.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv sur lequel appliquer la correction")
    parser.add_argument("-m", "--model", type=str, help="Nom du modèle")
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
    model_dir = f'ajouvenel/{model_name}'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Préparation du prompt et récupération de la réponse
    i = 0
    preds = []

    # Application de la correction au fichier
    ipython = get_ipython()
    for _ in tqdm(range(len(test))):

        prompt = f"""### Instruction:
            Fix the OCR errors in the provided text.
        
            ### Input:
            {test[i]['OCR Text']}
        
            ### Response:
            """
        
        input_ids = tokenizer(prompt, max_length=1024, return_tensors='pt', truncation=True).input_ids.to("cuda")
        with torch.inference_mode():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
        pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
        preds.append(pred)
        i += 1

    # Sauvegarde des résultats
    results = get_results(test, preds)
    results.to_csv(f'fine_tuned_models/result_{model_name}.csv', index=False)

    # Affichage des 10 premières lignes du fichier de résultats
    corrections = pd.read_csv(f'fine_tuned_models/result_{model_name}.csv')
    print(corrections.head(10))



if __name__ == "__main__":
    main()
