import cv2
import gc
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm

def resize_image(img, org_width, org_height, new_width, new_height):
    # Invert
    img = 255 - img

    # Normalize
    img = (img * (255.0 / img.max())).astype(np.uint8)

    # Reshape
    img = img.reshape(org_height, org_width)
    image_resized = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)

    return image_resized    

def resize_and_save_image(train_dir, img, org_width, org_height, new_width, new_height, image_id):
    # Resize Image
    image_resized = resize_image(img, org_width, org_height, new_width, new_height)

    # Save Image
    cv2.imwrite(train_dir + str(image_id) + '.png', image_resized)
        
def generate_images(data_dir, train_dir, org_width, org_height, new_width, new_height):
    for i in tqdm(range(0, 4)):

        # Read Parquet file
        df = pd.read_parquet(os.path.join(data_dir, 'train_image_data_'+str(i)+'.parquet'))
        # Get Image Id values
        image_ids = df['image_id'].values 
        # Drop Image_id column
        df = df.drop(['image_id'], axis = 1)
        
        # Loop over rows in Dataframe and generate images 
        for image_id, index in zip(image_ids, range(df.shape[0])):
            resize_and_save_image(train_dir,df.loc[df.index[index]].values, org_width, 
                                    org_height, new_width, new_height, image_id)
       
        # Cleanup
        del df
        gc.collect()