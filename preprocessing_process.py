from alive_progress import alive_bar
from preprocessing import Preprocessing
import pandas as pd
import time
from PIL import Image
import numpy as np

metadata = pd.read_csv("/Users/julia/Downloads/cancer-skin/metadata.csv")
folder_path = "/Users/julia/Downloads/cancer-skin/"

isic_id = list(metadata['isic_id'])
start_time = time.time()
with alive_bar(len(isic_id)) as bar:
    for i, img_name in enumerate(isic_id):
        #getting image
        img_path = folder_path + img_name + ".JPG"
        image = Image.open(img_path)
        data = np.asarray(image)
        #preprocessing image
        preprocessed = Preprocessing().perform(data, 480, 640)
        final = Image.fromarray(preprocessed).convert('RGB')
        output_path = '/Users/julia/Desktop/STUDIA/MGR_NEURO/II SEM/Chmura w ML/preprocessed/' + img_name + '.JPG'
        final.save(output_path)
        bar()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        elapsed_time_str = f"{elapsed_minutes:02d}:{elapsed_seconds:02d}"
        
        # Display elapsed time in the progress bar
        bar.text(f"Time elapsed: {elapsed_time_str}")

