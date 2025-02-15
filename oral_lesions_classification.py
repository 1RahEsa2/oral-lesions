import os

import numpy as np

import streamlit as st

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

from keras.metrics import Precision, Recall
from keras import backend as K

from PIL import Image



# Constant
IMG_HIGHT = 224
IMG_WIDTH = 224

CLASSES = {
    'CaS': 'Canker Sore',
    'CoS': 'Cold Sore',
    'Gum': 'Gum Disease',
    'MC': 'Mucosal Condition',
    'OC': 'Oral Cancer',
    'OLP': 'Oral Lichen Planus',
    'OT': 'Other'
}

MODEL_PATH = './densenet_model.keras'




#define custom layer
class SqueezeExcite(layers.Layer):
        def __init__(self, ratio=16, **kwargs):
            super(SqueezeExcite, self).__init__(**kwargs)
            self.ratio = ratio

        def build(self, input_shape):
            filters = input_shape[-1]
            self.global_avg_pool = layers.GlobalAveragePooling2D()
            self.dense1 = layers.Dense(filters // self.ratio, activation='relu', use_bias=False)
            self.dense2 = layers.Dense(filters, activation='sigmoid', use_bias=False)
            self.reshape = layers.Reshape((1, 1, filters))

        def call(self, inputs, training=False):
            se = self.global_avg_pool(inputs)
            se = self.dense1(se)
            se = self.dense2(se)
            se = self.reshape(se)
            return layers.multiply([inputs, se])

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', average='micro', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


            
@st.cache_resource
def loadModel(model_path:str, custom_objects: dict ):
    """
    this function will accept the model path
      and any custom layer you build for the model then load it 
    """
    try:
        
        model = load_model(model_path,
                          custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# callbacks
def preprocessing(image_path: str):
    """this func will preprocess the image 
    to make it compatabile with the image that the model train on """

    img = image.load_img(image_path,
                           target_size=(IMG_HIGHT, IMG_WIDTH))
    
    input_array = image.img_to_array(img)

    # Add batch dimension to be able to fed the model one photo
    input_array = np.expand_dims(input_array, axis=0)  

    normalize_image = input_array / 255.0  
    return normalize_image



st.title("Classify oral lesions")

# File uploader accepts video files
Image_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if Image_file is not None:

    Image_name = Image_file.name
    Image_name = os.path.splitext(os.path.basename(Image_name))[0]

    # Display image
    st.image(Image_file, caption="Uploaded Image", use_container_width=True)

    # Save the image
    img_path = f"{Image_name}.jpg"
    with open(img_path, "wb") as f:
        f.write(Image_file.getbuffer())

    st.success("Image uploaded successfully!")
    
    # Uplad the model
        
    model = loadModel(MODEL_PATH, custom_objects={"SqueezeExcite": SqueezeExcite,
                                                  'F1Score': F1Score})
    if model:
        # Preprocess image
        input_image = preprocessing(img_path)

        # Predict
        prediction = model.predict(input_image)
        classification = np.argmax(prediction)

        # Display result
        st.write(f"Predicted Class: {CLASSES.get(list(CLASSES.keys())[classification], 'Unknown')}")

    