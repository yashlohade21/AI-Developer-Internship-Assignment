from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def create_rcnn_model(input_shape, num_classes):
    # Use a pre-trained ResNet50 as the backbone for feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Create the Region-based CNN (R-CNN) model on top of the pre-trained ResNet backbone
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
