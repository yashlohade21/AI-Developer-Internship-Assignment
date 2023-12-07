from model import create_rcnn_model
from preprocess_dataset import preprocess_dataset
from tensorflow.keras.optimizers import Adam
import numpy as np

# Set your input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 20

# Create and compile the R-CNN model
model = create_rcnn_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Implement data loading based on your dataset structure
# Load preprocessed data (adjust the path accordingly)
images = np.load('processed_data/images.npy')
labels = np.load('processed_data/labels.npy')

# Train the model
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)
