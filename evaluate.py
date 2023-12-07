from model import create_rcnn_model
from preprocess_dataset import preprocess_dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Set your input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 20

# Create the R-CNN model
model = create_rcnn_model(input_shape, num_classes)

# Implement data loading for evaluation based on your dataset structure
# Load preprocessed data (adjust the path accordingly)
images = np.load('processed_data/images.npy')
labels = np.load('processed_data/labels.npy')

# Print debugging information
print("Number of samples in images:", len(images))
print("Number of samples in labels:", len(labels))

# Ensure the number of samples in images and labels match
if len(images) == len(labels):
    # Evaluate the model
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)

    # Print classification report and confusion matrix
    print("Classification Report:\n", classification_report(true_labels, predicted_labels))
    print("Confusion Matrix:\n", confusion_matrix(true_labels, predicted_labels))
else:
    print("Error: Number of samples in images and labels do not match.")
