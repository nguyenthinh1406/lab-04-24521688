import streamlit as st
import cv2
import numpy as np
import keras
from keras import layers, models
from keras.models import load_model
import tensorflow as tf
import time
import pandas as pd
# from transformers import AutoConfig, TFViTModel
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

augment = keras.Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(factor = 0.2),
    layers.RandomContrast(factor = 0.5)
], name = 'augment')

# class ViTClassifier(keras.Model):
#     def __init__(self, num_classes = 6, **kwargs):
#         super().__init__(**kwargs)
#         config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
#         self.vit_backbone = TFViTModel.from_pretrained(
#             'google/vit-base-patch16-224', config = config
#         )
#         self.vit_backbone.trainable = False
#         self.head = keras.Sequential([
#             layers.Dense(128, use_bias = False),
#             layers.BatchNormalization(),
#             layers.Activation('relu'),
#             layers.Dropout(0.5),
#             layers.Dense(num_classes, activation = 'softmax')
#         ], name = "classification_head")
#     def call(self, inputs):
#         outputs = self.vit_backbone(inputs) 
#         x = outputs.last_hidden_state[:, 0, :]
#         return self.head(x)

@st.cache_resource
def load_all_models():
    common_objects = {
        'augment': augment,
        'sequential': augment,
        'sequential_1': augment
        # 'ViTClassifier': ViTClassifier,
        # 'TFViTModel': TFViTModel
    }

    custom_objects_vgg = common_objects.copy()
    custom_objects_vgg['preprocess_input'] = vgg_preprocess
    model_vgg = load_model('vgg19_model.keras', custom_objects=custom_objects_vgg)
    st.success("Tải model VGG19 thành công!")

    custom_objects_resnet = common_objects.copy()
    custom_objects_resnet['preprocess_input'] = resnet_preprocess
    model_resnet = load_model('resnet50_model.keras', custom_objects=custom_objects_resnet)
    st.success("Tải model ResNet50 thành công!")

    custom_objects_mobile = common_objects.copy()
    custom_objects_mobile['preprocess_input'] = mobile_preprocess
    model_mobile = load_model('mobileV2_model.keras', custom_objects=custom_objects_mobile)
    st.success("Tải model MobileNetV2 thành công!")

    custom_objects_efficient = common_objects.copy()
    custom_objects_efficient['preprocess_input'] = efficient_preprocess
    model_efficient = load_model('efficientB0_model.keras', custom_objects=custom_objects_efficient)
    st.success("Tải model EfficientNetB0 thành công!")

    # model_vit = load_model('vitB16_model.keras', custom_objects=common_objects)
    # st.success("Tải model ViTB16 thành công!")
    
    return model_vgg, model_resnet, model_mobile, model_efficient


try:
    model_vgg, model_resnet, model_mobile, model_efficient = load_all_models()
    st.success("Đã tải tất cả 4 model!")
except Exception as e:
    st.error(f"Lỗi khi tải model: {e}")
    st.stop()
upload_im = st.file_uploader("Chọn ảnh của bạn", type=["png", "jpg", "jpeg"])

if upload_im is not None:
    img_original = np.asarray(bytearray(upload_im.read()), dtype = np.uint8)
    img_original = cv2.cvtColor(cv2.imdecode(img_original, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    st.image(img_original, caption='Ảnh đã tải lên.', use_column_width=True)
    
    model_list = [model_vgg, model_resnet, model_mobile, model_efficient]
    model_names = ['VGG19', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']
    preprocess_funcs = [vgg_preprocess, resnet_preprocess, mobile_preprocess, efficient_preprocess]
    class_name = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    predicted_class = []
    confidence_score = []
    inference_time = []

    for i in range(len(model_list)):
        model = model_list[i]
        name = model_names[i]
        preprocess_func = preprocess_funcs[i]

        # if name == 'ViTB16':
        #     target_size = (224, 224)
        #     img_resized = cv2.resize(img_original, target_size)
        #     img_batch = np.expand_dims(img_resized, axis=0)
        #     img_processed = img_batch.astype('float32') / 255.0
        # else:
        target_size = (256, 256)
        img_resized = cv2.resize(img_original, target_size)
        img_batch = np.expand_dims(img_resized, axis=0)
        img_processed = preprocess_func(img_batch.astype('float32'))
            
        start = time.time()
        pred = model.predict(img_processed)
        end = time.time()
        inference_time.append(end - start)
        predicted_class.append(class_name[np.argmax(pred)])
        confidence_score.append(np.max(pred)*100)

    df = pd.DataFrame({
        'Model': ['VGG19', 'ResNet50', 'MobileNetV2', 'EfficientNetB0'],
        'Predicted class': predicted_class,
        'Confidence(%)': confidence_score,
        'Inference time(s)': inference_time
    })

    st.dataframe(df)

    st.bar_chart(df, x = 'Model', y = 'Confidence(%)')

    st.bar_chart(df, x = 'Model', y = 'Inference time(s)')
