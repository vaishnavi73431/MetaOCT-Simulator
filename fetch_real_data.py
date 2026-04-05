import os
import json
import itertools
from datasets import load_dataset

def fetch_real_oct_images():
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Label mapping in the keremberke dataset
    labels_map = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
    
    print("Connecting to Hugging Face Cloud to stream real OCT images...)")
    print("This will only download ~5MB instead of 12GB!")
    
    try:
        # Streaming=True ensures we don't download the zip!
        dataset = load_dataset("keremberke/oct-image-classification", "full", split="train", streaming=True)
    except Exception as e:
        print(f"Error connecting to dataset: {e}")
        return

    ground_truth = {}
    counts = {"CNV": 0, "DME": 0, "DRUSEN": 0, "NORMAL": 0}
    target_per_class = 10 # 40 images total (10 per class)
    
    for item in dataset:
        label_id = item["labels"]
        label_name = labels_map.get(label_id, "NORMAL")
        
        if counts[label_name] < target_per_class:
            img = item["image"]
            filename = f"{label_name}_{counts[label_name] + 1}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save the image
            img.save(filepath)
            
            # Formulate the ground truth metadata
            keywords = []
            if label_name == "CNV": keywords = ["subretinal fluid", "rpe elevation", "neovascularization"]
            elif label_name == "DME": keywords = ["intraretinal cysts", "thickening", "edema"]
            elif label_name == "DRUSEN": keywords = ["rpe deposits", "drusen"]
            else: keywords = ["normal foveal contour", "intact rpe"]
            
            # Realistic mock bounding boxes for demonstration where pathology usually exists
            box = [[0, 0], [0, 0]]
            if label_name != "NORMAL":
                box = [[80, 80], [150, 150]] # Center of the macula where fluid usually is
            
            ground_truth[filename] = {
                "label": label_name,
                "box": box,
                "keywords": keywords
            }
            
            counts[label_name] += 1
            print(f"Downloaded {filename}...")
            
        # Break if we have exactly 10 of each
        if all(c >= target_per_class for c in counts.values()):
            break
            
    # Save the strictly formatted JSON
    with open("ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"\nSuccessfully downloaded {sum(counts.values())} real images!")
    print("Auto-generated the new ground_truth.json answer key.")
    print("You can now safely delete the 3 old 'sample_X.jpg' black images.")

if __name__ == "__main__":
    fetch_real_oct_images()
