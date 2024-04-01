import os
import cv2
import json
from figure import detect_subfigures, show_labels_on_image, detect_subfigure_labels

def create_annotation(image_path, bounding_boxes, final_path=None):
    final_path = final_path or image_path
    
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Create the annotation dictionary
    annotation = {
        "annotations": [
            {
                "result": []
            }
        ],
        "data": {
            "image": final_path
        }
    }

    # Iterate over the bounding boxes
    for box in bounding_boxes:
        x1, y1, bwidth, bheight = box[1:5]

        # Normalize the coordinates
        x = x1 / width * 100
        y = y1 / height * 100
        width_norm = bwidth / width * 100
        height_norm = bheight / height * 100

        # Add the bounding box to the annotation
        annotation["annotations"][0]["result"].append({
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": x,
                "y": y,
                "width": width_norm,
                "height": height_norm,
                "rotation": 0,
                "rectanglelabels": ["label"]
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual"
        })

    return annotation

def extract_paths(data):
    path_dict = {}
    for item in data:
        filename = os.path.basename(item['image']).split('-')[1]
        path_dict[filename] = item['image']
    return path_dict





if __name__ == "__main__":
    with open('uploaded.json') as f:
        data = json.load(f)
    path_dict = extract_paths(data)
    
    # img_path = 'imgs/10260401_Fig10.jpg'
    # img_name = os.path.basename(img_path)
    # bbox = detect_subfigures(img_path)
    # subfigure_info, concate_img = detect_subfigure_labels(img_path, bbox)
    # show_labels_on_image(img_path, subfigure_info)
    # print(create_annotation(img_path, subfigure_info, path_dict[img_name]))


    image_folder = "/Users/etowah/projects/PMC-figure-downloader/img"
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith((".png", ".jpg", ".gif"))]
    annots = []
    not_found = []
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}")
        img_name = os.path.basename(image_path)
        bbox = detect_subfigures(image_path)
        subfigure_info, concate_img = detect_subfigure_labels(image_path, bbox)
        if path_dict.get(img_name) is None:
            print(f"Image {img_name} not found in the uploaded data")
            not_found.append(img_name)
            continue
        annotation = create_annotation(image_path, subfigure_info, path_dict[img_name])
        annots.append(annotation)
    
    with open(f"annotations.json", "w") as f:
        json.dump(annots, f, indent=2)

    with open(f"not_found.json", "w") as f:
        json.dump(not_found, f, indent=2)