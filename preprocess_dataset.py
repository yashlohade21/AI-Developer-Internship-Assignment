import os
from xml.etree import ElementTree as ET
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import numpy as np


def parse_annotation(annotation_path):
    # """Parses an annotation file in PASCAL VOC format and extracts the object labels and bounding boxes."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    samples = []

    for obj in root.findall("object"):
        label = obj.find("name").text

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        boxes = [xmin, ymin, xmax, ymax]

        samples.append((label, boxes))

    return samples


def preprocess_dataset(data_folder, output_folder):
    # """
    # Preprocesses the entire dataset by loading images and annotations, extracting labels and bounding boxes,
    # converting labels to one-hot encoding, and saving preprocessed data as NumPy arrays.
    # """
    images = []
    class_names_list = []
    boxes_list = []

    images_folder = os.path.join(data_folder, "JPEGImages")
    annotations_folder = os.path.join(data_folder, "Annotations")

    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(images_folder, image_file)
            annotation_path = os.path.join(
                annotations_folder, os.path.splitext(image_file)[0] + ".xml"
            )

            # Check for missing annotation
            if not os.path.exists(annotation_path):
                raise ValueError(f"Missing annotation for image: {image_file}")

            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)

            images.append(img_array)

            samples = parse_annotation(annotation_path)
            class_names, boxes = zip(*samples)

            # Check for mismatched sample counts
            if len(class_names) != len(boxes):
                raise ValueError(
                    f"Mismatched sample counts for image: {image_file} (Class Names: {len(class_names)}, Boxes: {len(boxes)})"
                )

            class_names_list.append(class_names)
            boxes_list.append(boxes)

    images = np.array(images)

    # Flatten the lists of class names and boxes
    flat_class_names = [class_name for sublist in class_names_list for class_name in sublist]
    flat_boxes = [box for sublist in boxes_list for box in sublist]

    # Convert labels to one-hot encoding
    unique_labels = list(set(flat_class_names))
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    labels = [label_mapping[label] for label in flat_class_names]
    labels = to_categorical(labels, num_classes=len(unique_labels))

    # Save preprocessed data
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "images.npy"), images)
    np.save(os.path.join(output_folder, "labels.npy"), labels)


if __name__ == "__main__":
    data_folder = r"VOCdevkit/VOC2012"
    output_folder = "processed_data"  # Adjust this based on your desired output path

    preprocess_dataset(data_folder, output_folder)
