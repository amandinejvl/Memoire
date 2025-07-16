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

    # Suppression des arobases
    df = df.replace('@', '', regex=True)

    try:

        # Calcul du CER moyen
        df['cer'] = df.apply(lambda row: cer(row['gold_sent'], row['ocr_sent']), axis=1)
        mean_cer = df['cer'].mean()
        print(f"CER moyen : {mean_cer}")

        # Calcul du WER moyen
        df['wer'] = df.apply(lambda row: wer(row['gold_sent'], row['ocr_sent']), axis=1)
        mean_wer = df['wer'].mean()
        print(f"WER moyen : {mean_wer}")
    
    except KeyError:

        # Calcul du CER moyen
        df['cer'] = df.apply(lambda row: cer(row['Ground Truth'], row['OCR Text']), axis=1)
        mean_cer = df['cer'].mean()
        print(f"CER moyen : {mean_cer}")

        # Calcul du WER moyen
        df['wer'] = df.apply(lambda row: wer(row['Ground Truth'], row['OCR Text']), axis=1)
        mean_wer = df['wer'].mean()
        print(f"WER moyen : {mean_wer}")



if __name__ == '__main__':
    main()