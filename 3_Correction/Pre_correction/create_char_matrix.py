import pandas as pd
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity



def char_to_image(char, size=100, bg_color="white", text_color="black", font_path=None, blur_radius=2.5):
    """Transforme un caractère en image de ce caractère"""

    # Initialiser les paramètres de l'image
    image = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Appliquer la police choisie
    if font_path:
        font = ImageFont.truetype(font_path, int(size * 0.8))
    else:
        font = ImageFont.load_default()
    
    # Choisir les dimensions des boîtes
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) / 2 - bbox[0]
    y = (size - text_height) / 2 - bbox[1]
    
    # Créer l'image
    draw.text((x, y), char, fill=text_color, font=font)
    
    # Flouter l'image
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return image



def image_to_vector(img):
    """Convertir l'image en vecteur"""
    
    # Convertir l'image en niveaux de gris
    img_gray = img.convert('L')
    
    # Redimensionner l'image pour qu'elles aient toutes les mêmes dimensions (au cas où)
    img_resized = img_gray.resize((100, 100))

    # Convertir en tableau numpy
    arr = np.array(img_resized)

    # Aplatir le tableau en 1 dimension
    return arr.flatten()



def save_matrix_with_labels(matrix, labels):
    """Sauvegarde la matrice avec ses labels dans un fichier csv"""
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv("char_sim_matrix.csv", encoding="utf-8")
    print("La matrice de similarité visuelle des caractères a bien été sauvegardée dans le fichier char_sim_matrix.csv.")



def main():

    # Parsing de l'argument
    parser = argparse.ArgumentParser(description='Crée une matrice de similarité visuelles des caractères.')
    parser.add_argument("-f", "--file", type=str, help="Chemin vers le fichier csv contenant l'ensemble d'entraînement'")
    args = parser.parse_args()

    # Charger l'ensemble d'entraînement
    df = pd.read_csv(args.file)

    # Extraire tous les caractères présents dans les phrases OCR et gold
    ocr = ''.join(df['OCR Text'])
    gold = ''.join(df['Ground Truth'])
    all_chars = ocr + gold
    unique_chars = sorted(set(all_chars))

    # Choisir la police à utiliser pour la génération des images
    font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    
    # Générer les images de tous les caractères
    char_images = []
    for char in unique_chars:
        img = char_to_image(char, font_path=font_path)
        #img.show()
        char_images.append((char, img))

    # Transformer les images en vecteurs
    chars = [couple[0] for couple in char_images]
    vectors = [image_to_vector(couple[1]) for couple in char_images]
    vectors = np.array(vectors)

    # Créer la matrice de similarité cosinus
    similarity_matrix = cosine_similarity(vectors)

    # Visualiser la matrice
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, xticklabels=chars, yticklabels=chars, annot=False, cmap="Reds")
    plt.xticks(fontsize=6, rotation=0)
    plt.yticks(fontsize=6)
    plt.title("Matrice de similarité visuelle entre les caractères")
    plt.show()

    # Sauvegarder la matrice de similarité
    save_matrix_with_labels(similarity_matrix, chars)



if __name__ == "__main__":
    main()