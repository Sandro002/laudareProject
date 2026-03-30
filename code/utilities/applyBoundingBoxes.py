from tqdm import tqdm
from PIL import Image
import os


def applyBoundingBox(pages_dict, img_folder):

    print("APPLICANDO BOUNDING BOX")
    for page in tqdm(pages_dict.values()): 
        
        img_filename = page.filename
        image_path = os.path.join(img_folder, img_filename)

        if not os.path.exists(image_path):
            print(f"Non trovato: {image_path}")
            continue
        
        with Image.open(image_path) as img:
            
            for cat in page.annotations:
                folder_name = None
                
                if cat.category == 1:
                    clean_desc = cat.desc.replace('(', '').replace(')', '').strip()
                    num_notes = len(clean_desc.split()) if clean_desc else 0
                    if num_notes > 0 and num_notes >= 4:
                        folder_name = f"neume_4Plus"
                    if num_notes > 0 and num_notes < 4:
                        folder_name = f"neume_{num_notes}"
                
                elif cat.category == 2:
                    if len(cat.desc) >= 2:
                        clef_type = cat.desc[1]
                        folder_name = f"clef_{clef_type}"
                
                elif cat.category == 3:
                    folder_name = "custos"
                
                elif cat.category == 6:
                    folder_name = "lines"
                
                if folder_name:
                    os.makedirs(f"dataset/{folder_name}", exist_ok=True)
                    
                    left = cat.x
                    top = cat.y
                    right = cat.x + cat.w
                    bottom = cat.y + cat.h

                    crop = img.crop((left, top, right, bottom))

                    if crop.size[0] == 0 or crop.size[1] == 0:
                        continue

                    save = f"dataset/{folder_name}/{img_filename.split('.')[0]}_id{cat.id}.png"
                    crop.save(save, "PNG")

    print('\nBOUNDING BOX APPLICATE CON SUCCESSO')


if __name__ == "__main__":
    
    applyBoundingBox("../../I-Fn_BR_18", "../../dataset")