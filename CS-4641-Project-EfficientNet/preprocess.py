import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# function to center onto the retinal area from the whole image
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if mask.any():
            img = img[np.ix_(mask.any(1), mask.any(0))]
        return img

def preprocess_images(input_dir, output_dir, csv_file, target_size=(224, 224)): #edit target size for dimensions
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['id_code'] + '.png'
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, target_size)
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    preprocess_images('train_images', 'preprocessed_train_images', 'train.csv')
    print("Preprocessing complete. Images saved to 'preprocessed_train_images/'.")