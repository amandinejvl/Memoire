import pandas as pd
import argparse
from jiwer import wer, cer



def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description="Calcule le CER moyen entre les phrases gold et OCR à partir d'un fichier csv.")
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv contenant les paires de phrases gold et OCR")
    args = parser.parse_args()

    # Lecture du fichier csv
    df = pd.read_csv(args.file)

    # Suppression des valeurs NaN
    df = df.fillna('')

    # Extraire les phrases gold
    gold_sentences = df['Ground Truth']

    # Extraire les phrases OCR
    ocr_sentences = df['OCR Text']

    # Extraire les phrases pré-corrigées
    lex_sentences = df['Lexical Correction']

    # Extraire les phrases corrigées
    corr_sentences = df['Model Correction']

    n = 0
    for gold, ocr, lex, corr in zip(gold_sentences, ocr_sentences, lex_sentences, corr_sentences):
        if corr != gold:
            print(f"gold : {gold}")
            print(f"OCR  : {ocr}")
            print(f"lex  : {lex}")
            print(f"corr : {corr}")
            print(f"WER  : {wer(gold, corr)}")
            print("_"*80)
        else:
            n += 1
    print(f"{n} phrases rétablies")
    
    # Calcul du CER moyen
    df['cer'] = df.apply(lambda row: cer(row['Ground Truth'], row['Lexical Correction']), axis=1)
    mean_cer = df['cer'].mean()
    print(f"CER moyen : {mean_cer}")

    # Calcul du WER moyen
    df['wer'] = df.apply(lambda row: wer(row['Ground Truth'], row['Lexical Correction']), axis=1)
    mean_wer = df['wer'].mean()
    print(f"WER moyen : {mean_wer}")



    '''old_error = 0.311
    new_error = 0.271
    reduction_percent = ((old_error - new_error) / old_error) * 100
    print(reduction_percent)'''


    




if __name__ == '__main__':
    main() 
 
