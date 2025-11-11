import streamlit as st
import cv2
import numpy as np
import keras
from keras import layers, models
from keras.models import load_model
import tensorflow as tf
import time
import pandas as pd
from PIL import Image
import timm
import torch.nn as nn
import torch
import torchvision.transforms as T

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

augment = keras.Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(factor = 0.2),
    layers.RandomContrast(factor = 0.5)
], name = 'augment')

vit_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_all_models():
    common_objects = {
        'augment': augment,
        'sequential': augment,
        'sequential_1': augment
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

    model_vit = timm.create_model(
        'vit_base_patch16_224', 
        pretrained = False, 
    )      

    in_features_vit = model_vit.head.in_features
    model_vit.head = nn.Linear(in_features_vit, 6)

    model_vit.load_state_dict(torch.load('model_vit.pth'))
    model_vit.eval()
    st.success("Tải model ViTB16 thành công!")
    
    return model_vgg, model_resnet, model_mobile, model_efficient, model_vit


try:
    model_vgg, model_resnet, model_mobile, model_efficient, model_vit = load_all_models()
    st.success("Đã tải tất cả 5 model!")
except Exception as e:
    st.error(f"Lỗi khi tải model: {e}")
    st.stop()
upload_im = st.file_uploader("Chọn ảnh của bạn", type=["png", "jpg", "jpeg"])

if upload_im is not None:
    img_original = np.asarray(bytearray(upload_im.read()), dtype = np.uint8)
    img_original = cv2.cvtColor(cv2.imdecode(img_original, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_original)
    
    st.image(img_original, caption = 'Ảnh đã tải lên.', use_column_width = True)
    
    model_list = [model_vgg, model_resnet, model_mobile, model_efficient, model_vit]
    model_names = ['VGG19', 'ResNet50', 'MobileNetV2', 'EfficientNetB0', 'ViTNetB16']
    preprocess_funcs = [vgg_preprocess, resnet_preprocess, mobile_preprocess, efficient_preprocess, None]
    class_name = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    predicted_class = []
    confidence_score = []
    inference_time = []

    for model, name, preprocess_func in zip(model_list, model_names, preprocess_funcs):
        st.write(f"--- \nĐang xử lý với model: **{name}**")

        if name == 'ViTB16':
            start = time.time()
            img_tensor = vit_transform(img_pil).unsqueeze(0)
            with torch.no_grad():
                logits = model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            conf = probabilities.max().item() * 100
            pred_idx = probabilities.argmax().item()
            pred_class = class_name[pred_idx]
            end = time.time()

            inference_time.append(end - start)
            predicted_class.append(pred_class)
            confidence_score.append(conf)

        else:
            target_size = (256, 256)
            img_resized = cv2.resize(img_original, target_size)
            img_batch = np.expand_dims(img_resized, axis=0)
            img_processed = preprocess_func(img_batch.astype('float32'))

            start = time.time()
            pred = model.predict(img_processed)
            end = time.time()

            inference_time.append(end - start)
            predicted_class.append(class_name[np.argmax(pred)])
            confidence_score.append(np.max(pred) * 100)

    df = pd.DataFrame({
        'Model': model_names,
        'Predicted class': predicted_class,
        'Confidence(%)': confidence_score,
        'Inference time(s)': inference_time
    })

    st.dataframe(df)

    st.bar_chart(df, x = 'Model', y = 'Confidence(%)')

    st.bar_chart(df, x = 'Model', y = 'Inference time(s)')
