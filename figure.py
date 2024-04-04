from figures.models.yolov3 import YOLOv3
from figures.models.yolov3 import YOLOv3img
from figures.models.network import resnet152
import cv2
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch
import process
import yaml
from skimage import io
import torch.nn.functional as F
from scipy.special import softmax
from subfigure_label_detection import detect_subfigure_labels
import os

def detect_subfigures(fig_path: str):
    configuration_file = "figures/config/yolov3_default_subfig.cfg"
    with open(configuration_file, "r") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    image_size = configuration["TEST"]["IMGSIZE"]
    nms_threshold = configuration["TEST"]["NMSTHRE"]
    confidence_threshold = 0.2

    model = YOLOv3(configuration["MODEL"])
    checkpoint = "weights/object_detection_model.pt"
    device = torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    img = io.imread(fig_path)
    if len(np.shape(img)) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img, info_img = process.preprocess(img, image_size, jitter=0)
    img = np.transpose(img / 255.0, (2, 0, 1))
    img = np.copy(img)
    img = torch.from_numpy(img).float().unsqueeze(0)
    img = Variable(img.type(torch.FloatTensor))

    img_raw = Image.open(fig_path).convert("RGB")
    width, height = img_raw.size

    ## Run model on figure
    with torch.no_grad():
        outputs = model(img.to(device))
        outputs = process.postprocess(
            outputs,
            dtype=torch.FloatTensor,
            conf_thre=confidence_threshold,
            nms_thre=nms_threshold,
        )

    ## Reformat model outputs to display bounding boxes in our desired format
    ## List of lists where each inner list is [x1, y1, x2, y2, confidence]
    subfigure_info = list()

    if outputs[0] is None:
        print("No Objects Detected! in {}".format(fig_path))
        return subfigure_info

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
        box = process.yolobox2label(
            [
                y1.data.cpu().numpy(),
                x1.data.cpu().numpy(),
                y2.data.cpu().numpy(),
                x2.data.cpu().numpy(),
            ],
            info_img,
        )
        box[0] = int(min(max(box[0], 0), width - 1))
        box[1] = int(min(max(box[1], 0), height - 1))
        box[2] = int(min(max(box[2], 0), width))
        box[3] = int(min(max(box[3], 0), height))
        # ensures no extremely small (likely incorrect) boxes are counted
        small_box_threshold = 5
        if (
            box[2] - box[0] > small_box_threshold
            and box[3] - box[1] > small_box_threshold
        ):
            box.append("%.3f" % (cls_conf.item()))
            subfigure_info.append(box)
    return subfigure_info

def dummy_detect_label_letter(figure_path, subfigure_info):
    img_raw = Image.open(figure_path).convert("RGB")
    img_raw = img_raw.copy()
    width, height = img_raw.size
    binary_img = np.zeros((height, width, 1))

    detected_labels = []
    detected_bboxes = []
    for subfigure in subfigure_info:
        ## Preprocess the image for the model
        x1, y1, x2, y2 = subfigure[:4]
        
        detected_labels.append("a")
        detected_bboxes.append([1, x1, y1, x2, y2])
    assert len(detected_labels) == len(detected_bboxes)

    ## subfigure_info (list of tuples): [(x1, y1, x2, y2, label)
    ##  where x1, y1 are upper left x and y coord divided by image width/height
    ##  and label is the an integer n meaning the label is the nth letter
    subfigure_info = []
    for i, label_value in enumerate(detected_labels):
        conf, x1, y1, x2, y2 = detected_bboxes[i]
        if (x2 - x1) < 64 and (
            y2 - y1
        ) < 64:  # Made this bigger because it was missing some images with labels
            binary_img[y1:y2, x1:x2] = 255
            label = ord(label_value) - ord("a")
            subfigure_info.append(
                (label, float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            )
    # concate_img needed for classify_subfigures
    concate_img = np.concatenate((np.array(img_raw), binary_img), axis=2)

    return subfigure_info, concate_img


def detect_label_letter(figure_path, subfigure_info):
    """Uses text recognition to read subfigure labels from figure_path

    Note:
        To get sensible results, should be run only after
        detect_subfigure_boundaries has been run
    Args:
        figure_path (str): A path to the image (.png, .jpg, or .gif)
            file containing the article figure
        subfigure_info (list of lists): Details about bounding boxes
            of each subfigure from detect_subfigure_boundaries(). Each
            inner list has format [x1, y1, x2, y2, confidence] where
            x1, y1 are upper left bounding box coordinates as ints,
            x2, y2, are lower right, and confidence the models confidence
    Returns:
        subfigure_info (list of tuples): Details about bounding boxes and
            labels of each subfigure in figure. Tuples for each subfigure are
            (x1, y1, x2, y2, label) where x1, y1 are upper left x and y coord
            divided by image width/height and label is the an integer n
            meaning the label is the nth letter
        concate_img (np.ndarray): A numpy array representing the figure.
            Used in classify_subfigures. Ideally this will be removed to
            increase modularity.
    """
    model = resnet152()
    checkpoint = "weights/text_recognition_model.pt"
    device = torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    img_raw = Image.open(figure_path).convert("RGB")
    img_raw = img_raw.copy()
    width, height = img_raw.size
    binary_img = np.zeros((height, width, 1))

    detected_labels = []
    detected_bboxes = []
    for subfigure in subfigure_info:
        ## Preprocess the image for the model
        bbox = tuple(subfigure[:4])
        img_patch = img_raw.crop(bbox)
        img_patch = np.array(img_patch)[:, :, ::-1]
        img_patch, _ = process.preprocess(img_patch, 28, jitter=0)
        img_patch = np.transpose(img_patch / 255.0, (2, 0, 1))
        img_patch = torch.from_numpy(img_patch).type(torch.FloatTensor).unsqueeze(0)

        ## Run model on figure
        label_prediction = model(img_patch.to(device))
        label_confidence = np.amax(
            F.softmax(label_prediction, dim=1).data.cpu().numpy()
        )
        label_value = chr(
            label_prediction.argmax(dim=1).data.cpu().numpy()[0] + ord("a")
        )
        if label_value == "z":
            continue

        ## Reformat results for to desired format
        x1, y1, x2, y2, box_confidence = subfigure
        total_confidence = float(box_confidence) * label_confidence
        if label_value in detected_labels:
            label_index = detected_labels.index(label_value)
            if total_confidence > detected_bboxes[label_index][0]:
                detected_bboxes[label_index] = [total_confidence, x1, y1, x2, y2]
        else:
            detected_labels.append(label_value)
            detected_bboxes.append([total_confidence, x1, y1, x2, y2])
    assert len(detected_labels) == len(detected_bboxes)

    ## subfigure_info (list of tuples): [(x1, y1, x2, y2, label)
    ##  where x1, y1 are upper left x and y coord divided by image width/height
    ##  and label is the an integer n meaning the label is the nth letter
    subfigure_info = []
    for i, label_value in enumerate(detected_labels):
        if (ord(label_value) - ord("a")) >= (len(detected_labels) + 2):
            continue
        conf, x1, y1, x2, y2 = detected_bboxes[i]
        if (x2 - x1) < 64 and (
            y2 - y1
        ) < 64:  # Made this bigger because it was missing some images with labels
            binary_img[y1:y2, x1:x2] = 255
            label = ord(label_value) - ord("a")
            subfigure_info.append(
                (label, float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            )
    # concate_img needed for classify_subfigures
    concate_img = np.concatenate((np.array(img_raw), binary_img), axis=2)

    return subfigure_info, concate_img


def classify_subfigures(figure_path, subfigure_labels, concate_img):
    """Classifies the type of image each subfigure in figure_path

    Note:
        To get sensible results, should be run only after
        detect_subfigure_boundaries and detect_subfigure_labels have run
    Args:
        figure_path (str): A path to the image (.png, .jpg, or .gif)
            file containing the article figure
        subfigure_labels (list of tuples): Information about each subfigure.
            Each tuple represents a single subfigure in the figure_path
            figure. The tuples are (label, x, y, width, height) where
            label is the n for the nth letter in the alphabet and x, y,
            width, and height are percentages of the image width and height
        concate_img (np.ndarray): A numpy array representing the figure.
            Has been modified in detect_subfigure_labels. Ideally this
            parameter will be removed to increase modularity.
    Returns:
        figure_json (dict): A figure json describing the data collected.
    Modifies:
        self.exsclaim_json (dict): Adds figure_json to exsclaim_json
    """
    configuration_file = "figures/config/yolov3_default_master.cfg"
    with open(configuration_file, "r") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    img_size = configuration["TEST"]["IMGSIZE"]

    model = YOLOv3img(configuration["MODEL"])
    checkpoint = "weights/classifier_model.pt"
    device = torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    label_names = [
        "background",
        "microscopy",
        "parent",
        "graph",
        "illustration",
        "diffraction",
        "basic_photo",
        "unclear",
        "OtherSubfigure",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]
    img = concate_img[..., :3].copy()
    mask = concate_img[..., 3:].copy()

    img, info_img = process.preprocess(img, img_size, jitter=0)
    img = np.transpose(img / 255.0, (2, 0, 1))
    mask = process.preprocess_mask(mask, img_size, info_img)
    mask = np.transpose(mask / 255.0, (2, 0, 1))
    new_concate_img = np.concatenate((img, mask), axis=0)
    img = torch.from_numpy(new_concate_img).float().unsqueeze(0)
    img = Variable(img.type(torch.FloatTensor))

    subfigure_labels_copy = subfigure_labels.copy()

    subfigure_padded_labels = np.zeros((80, 5))
    if len(subfigure_labels) > 0:
        subfigure_labels = np.stack(subfigure_labels)
        # convert coco labels to yolo
        subfigure_labels = process.label2yolobox(
            subfigure_labels, info_img, img_size, lrflip=False
        )
        # make the beginning of subfigure_padded_labels subfigure_labels
        subfigure_padded_labels[: len(subfigure_labels)] = subfigure_labels[:80]
    # conver labels to tensor and add dimension
    subfigure_padded_labels = (torch.from_numpy(subfigure_padded_labels)).unsqueeze(0)
    subfigure_padded_labels = Variable(subfigure_padded_labels.type(torch.FloatTensor))
    padded_label_list = [None, subfigure_padded_labels]
    assert subfigure_padded_labels.size()[0] == 1

    # prediction
    with torch.no_grad():
        outputs = model(img.to(device), padded_label_list)

    # select the 13x13 grid as feature map
    feature_size = [13, 26, 52]
    feature_index = 0
    preds = outputs[feature_index]
    preds = preds[0].data.cpu().numpy()

    ## Documentation
    figure_name = figure_path
    figure_json = {"master_images": []}
    figure_json["figure_name"] = figure_name
    figure_json.get("master_images", [])
    # create an unassigned field with a master images field if it doesn't exist
    figure_json.get("unassigned", {"master_images": []}).get("master_images", [])

    full_figure_is_master = True if len(subfigure_labels) == 0 else False

    # max to handle case where pair info has only 1 (the full figure is the master image)
    for subfigure_id in range(0, max(len(subfigure_labels), 1)):
        sub_cat, x, y, w, h = (
            (subfigure_padded_labels[0, subfigure_id] * 13)
            .to(torch.int16)
            .data.cpu()
            .numpy()
        )
        best_anchor = np.argmax(preds[:, y, x, 4])
        tx, ty = np.array(preds[best_anchor, y, x, :2] / 32, np.int32)
        best_anchor = np.argmax(preds[:, ty, tx, 4])
        x, y, w, h = preds[best_anchor, ty, tx, :4]
        classification = np.argmax(preds[best_anchor, int(ty), int(tx), 5:])
        master_label = label_names[classification]
        subfigure_label = chr(int(sub_cat / feature_size[feature_index]) + ord("a"))
        master_cls_conf = max(softmax(preds[best_anchor, int(ty), int(tx), 5:]))

        if full_figure_is_master:
            img_raw = Image.fromarray(np.uint8(concate_img[..., :3].copy()[..., ::-1]))
            x1 = 0
            x2 = np.shape(img_raw)[1]
            y1 = 0
            y2 = np.shape(img_raw)[0]
            subfigure_label = "0"

        else:
            x1 = x - w / 2
            x2 = x + w / 2
            y1 = y - h / 2
            y2 = y + h / 2

            x1, y1, x2, y2 = process.yolobox2label([y1, x1, y2, x2], info_img)

        ## Saving the data into a json. Eventually it would be good to make the json
        ## be updated in each model's function. This could eliminate the need to pass
        ## arguments from function to function. Currently the coordinates in
        ## subfigure_info are different from those output from classifier model. Also
        ## concate_image depends on operations performed in detect_subfigure_labels()
        master_image_info = {}
        master_image_info["classification"] = master_label
        master_image_info["confidence"] = float("{0:.4f}".format(master_cls_conf))
        master_image_info["height"] = y2 - y1
        master_image_info["width"] = x2 - x1
        master_image_info["geometry"] = []
        for x in [int(x1), int(x2)]:
            for y in [int(y1), int(y2)]:
                geometry = {}
                geometry["x"] = x
                geometry["y"] = y
                master_image_info["geometry"].append(geometry)
        subfigure_label_info = {}
        subfigure_label_info["text"] = subfigure_label
        subfigure_label_info["geometry"] = []

        if not full_figure_is_master:
            _, x1, y1, x2, y2 = subfigure_labels_copy[subfigure_id]
            x2 += x1
            y2 += y1
            for x in [int(x1), int(x2)]:
                for y in [int(y1), int(y2)]:
                    geometry = {"x": x, "y": y}
                    subfigure_label_info["geometry"].append(geometry)
        master_image_info["subfigure_label"] = subfigure_label_info
        figure_json.get("master_images", []).append(master_image_info)

    return figure_json


def show_image_with_bboxes(image_path, bboxes):
    # Load the image
    image = cv2.imread(image_path)

    # Loop through the bounding boxes and draw them on the image
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def subfigure_rule_extraction(image_path, bboxes):
    # Load the image
    image = cv2.imread(image_path)

    rows = []

    # Loop through the bounding boxes and draw them on the image
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_with_bboxes2(image_path, bboxes, save=False, show=True):
    # Load the image
    image = cv2.imread(image_path)

    # Loop through the bounding boxes and draw them on the image
    for entry in bboxes["master_images"]:
        bbox = entry["geometry"]
        x1 = bbox[0]["x"]
        y1 = bbox[0]["y"]
        x2 = bbox[3]["x"]
        y2 = bbox[3]["y"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if (save):
        figure_name = image_path.split("/")[-1].split(".")[0]
        cv2.imwrite(f"test/bbox_{figure_name}a.jpg", image)
    if (show):
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_subfigures(image_path, bboxes, out_dir="test"):
    img = cv2.imread(image_path)
    figure_name = image_path.split("/")[-1].split(".")[0]

    for i, entry in enumerate(bboxes["master_images"]):
        bbox = entry["geometry"]
        x1 = bbox[0]["x"]
        y1 = bbox[0]["y"]
        x2 = bbox[3]["x"]
        y2 = bbox[3]["y"]
        subfigure = img[y1:y2, x1:x2]
        cv2.imwrite(f"{out_dir}/{figure_name}_{i}.jpg", subfigure)


def show_labels_on_image(image_path: str, subfigure_info, save=False, show=True):
    image = cv2.imread(image_path)
    figure_name = image_path.split("/")[-1].split(".")[0]

    for subfigure in subfigure_info:
        label = chr(subfigure[0] + ord("a"))
        x1, y1, width, height = [int(x) for x in subfigure[1:5]]
        cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
        x2 = x1 + width + 10
        y2 = y1 + height
        cv2.putText(
            image,
            label,
            (x2, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if (save):
        cv2.imwrite(f"test/{figure_name}.jpg", image)
    if (show):
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == "__main__":
    # get all images in imgs folder
    image_folder = "imgs"
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith((".png", ".jpg", ".gif"))]

    image_files = ["imgs/10335933_Fig5.jpg"]
    for image_file in image_files:
        # bounding_boxes = detect_subfigures(image_file)
        # print(bounding_boxes)
        bounding_boxes = detect_subfigure_labels(image_file)
        # show_image_with_bboxes(image_file, bounding_boxes)
        subfigure_info, concate_img = dummy_detect_label_letter(image_file, bounding_boxes)
        # show_labels_on_image(image_file, subfigure_info, save=False, show=True)
        figure_json = classify_subfigures(image_file, subfigure_info, concate_img)
        save_subfigures(image_file, figure_json, out_dir="test")
        # show_image_with_bboxes2(image_file, figure_json, save=False, show=True)
