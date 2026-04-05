import os
import json
import numpy as np
from PIL import Image

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    print("MedMNIST not installed.")
    exit(1)

def fetch_retinamnist():
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading massive 224x224 RetinaMNIST (OCT Dataset) natively via NumPy...")
    
    info = INFO['retinamnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    # Download the 224x224 dataset split
    try:
        dataset = DataClass(split='train', download=True, size=224)
    except Exception as e:
        print(f"Error fetching MedMNIST: {e}")
        return

    images = dataset.imgs
    labels = dataset.labels.flatten()
    
    label_map = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
    counts = {"CNV": 0, "DME": 0, "DRUSEN": 0, "NORMAL": 0}
    target = 10
    
    ground_truth = {}
    
    for i in range(len(labels)):
        lbl_idx = labels[i]
        label_name = label_map.get(lbl_idx, "NORMAL")
        
        if counts[label_name] < target:
            # MedMNIST images are numpy arrays
            img_array = images[i]
            
            # The images are grayscale or RGB depending on dataset. Usually RGB for 224.
            if len(img_array.shape) == 2:
                img = Image.fromarray(img_array).convert("RGB")
            else:
                img = Image.fromarray(img_array).convert("RGB")
                
            filename = f"medmnist_{label_name}_{counts[label_name] + 1}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            img.save(filepath)
            
            keywords = []
            if label_name == "CNV": keywords = ["subretinal fluid", "rpe elevation", "neovascularization"]
            elif label_name == "DME": keywords = ["intraretinal cysts", "thickening", "edema"]
            elif label_name == "DRUSEN": keywords = ["rpe deposits", "drusen"]
            else: keywords = ["normal foveal contour", "intact rpe"]
            
            # Dynamic mock box
            box = [[0, 0], [0, 0]]
            if label_name != "NORMAL":
                box = [[80, 80], [150, 150]]
                
            ground_truth[filename] = {
                "label": label_name,
                "box": box,
                "keywords": keywords
            }
            
            counts[label_name] += 1
            print(f"Saved {filename}")
            
        if all(c >= target for c in counts.values()):
            break
            
    with open("ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"\nSuccessfully generated {sum(counts.values())} real medical JPGs!")
    print("Auto-generated the new ground_truth.json!")

if __name__ == "__main__":
    fetch_retinamnist()
