import pandas as pd
import argparse
import string
import matplotlib.pyplot as plt



def filtrer_matrice(df, k=20):

    # Garder uniquement les index/colonnes qui sont une lettre
    #valid_labels = [label for label in df.index if str(label).isalpha() and str(label) not in ["é"]]
    liste = ["e", "a", "u", "i", "l", "c", "o",  "n", "I", "T", "d", "b", "h", "t", "r"]
    sorted_liste = sorted(liste)
    valid_labels = [label for label in df.index if str(label) in sorted_liste]

    # Extraire la sous-matrice correspondante
    df_letters_only = df.loc[valid_labels, valid_labels]

    # Calculer la somme de chaque ligne (identique à celle des colonnes pour une matrice symétrique)
    sums = df.sum(axis=1)

    # Obtenir les k index avec les plus grandes sommes
    topk_indices = sums.nlargest(k).index

    # Extraire la sous-matrice correspondante
    df_topk = df.loc[topk_indices, topk_indices]

    # Puis, parmi celles-là, garder les k lettres les plus "actives"
    topk = df_letters_only.sum(axis=1).nlargest(k).index
    df_topk_letters = df_letters_only.loc[topk, topk]
    df_topk_letters = df_topk_letters.sort_index(axis=0).sort_index(axis=1)


    return df_topk_letters



def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description="Applique la correction avec un modèle et calcule l'amélioration.")
    parser.add_argument("-m", "--matrix", type=str, help="Chemin vers le fichier csv contenant la matrice de confusion des caractères")
    args = parser.parse_args()

    # Charger la matrice de confusion
    df = pd.read_csv(args.matrix, index_col=0)

    # Conserver uniquement les confusions les plus fréquentes
    filtered_df = filtrer_matrice(df)
    print(filtered_df)


    
    

    # Displaying dataframe as an heatmap
    # with diverging colourmap as RdYlBu
    plt.imshow(filtered_df, cmap ="Reds")

    # Displaying a color bar to understand
    # which color represents which range of data
    plt.colorbar()

    # Assigning labels of x-axis 
    # according to dataframe
    plt.xticks(range(len(filtered_df)), filtered_df.columns)

    # Assigning labels of y-axis 
    # according to dataframe
    plt.yticks(range(len(filtered_df)), filtered_df.index)

    # Displaying the figure
    plt.show()

    



if __name__ == "__main__":
    main()