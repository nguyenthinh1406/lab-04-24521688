import streamlit as st
import cv2
import numpy as np
import keras
from keras import layers, models
from keras.models import load_model
import tensorflow as tf
import time
import pandas as pd
from transformers import AutoConfig, TFViTModel

# model_vgg = load_model('vgg19_model.keras')
# model_resnet = load_model('resnet50_model.keras')
model_mobile = load_model('mobileV2_model.keras')
model_efficient = load_model('efficientB0_model.keras')

class ViTClassifier(keras.Model):
    def __init__(self, num_classes=6, **kwargs):
        super().__init__(**kwargs)
        
        config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
    
        self.vit_backbone = TFViTModel.from_pretrained(
            'google/vit-base-patch16-224', config=config
        )
        self.vit_backbone.trainable = False
        
        self.head = keras.Sequential([
            layers.Dense(128, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name="classification_head")

    def call(self, inputs):
        outputs = self.vit_backbone(inputs) 
        x = outputs.last_hidden_state[:, 0, :]
        return self.head(x)
    
model_vit = load_model(
    'vitB16_model.keras',
    custom_objects={'ViTClassifier': ViTClassifier} 
)

# upload_im = st.file_uploader("Chọn ảnh của bạn", type=["png", "jpg", "jpeg"])

# if upload_im is not None:
#     uploaded_im = np.asarray(bytearray(upload_im.read()), dtype=np.uint8)
#     uploaded_im = cv2.cvtColor(cv2.imdecode(uploaded_im, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#     uploaded_im = cv2.resize(uploaded_im, (256, 256))
#     img_batch = np.expand_dims(uploaded_im, axis=0)

#     model_list = [model_vgg, model_resnet, model_mobile, model_efficient, model_vit]
#     class_name = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

#     predicted_class = []
#     confidence_score = []
#     inference_time = []
#     for model in model_list:
#         start = time.time()
#         pred = model.predict(img_batch)
#         end = time.time()
#         inference_time.append(end - start)
#         predicted_class.append(class_name[np.argmax(pred)])
#         confidence_score.append(np.max(pred)*100)

#     df = pd.DataFrame({
#         'Model': ['VGG19', 'ResNet50', 'MobileNetV2', 'EfficientNetB0','ViTB16'],
#         'Predicted class': predicted_class,
#         'Confidence(%)': confidence_score,
#         'Inference time(s)': inference_time
#     })

#     st.dataframe(df)

#     st.bar_chart(df, x = 'Model', y = 'Confidence(%)')

#     st.bar_chart(df, x = 'Model', y = 'Inference time(s)')
