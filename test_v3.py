# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 22:57:03 2023

@author: python
"""
from st_files_connection import FilesConnection
import copy
import streamlit as st
from streamlit_option_menu import option_menu
import PIL
from colorthief import ColorThief
import webcolors
from google.cloud import storage
import os
import io
import tempfile
from ultralytics import YOLO
from tensorflow.keras.applications.resnet_rs import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import random
import requests
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread
import textwrap
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import gridspec
import tensorflow_hub as tf_hub
from streamlit_extras.let_it_rain import rain
from tensorflow.keras.applications.vgg16 import preprocess_input as pinp
from io import BytesIO
import streamlit.components.v1 as components
from st_audiorec import st_audiorec
import speech_recognition as SR
import deepl 
import openai
import base64
from PIL import Image
import streamlit as st
import replicate
from decouple import config
from gtts import gTTS
import time
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import sentence_transformers as strd
from streamlit_modal import Modal
import sys
import urllib.request
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import sqlite3
from selenium.webdriver.common.keys import Keys
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

urls = [('https://storage.googleapis.com/csac_final_v2/new/model_resnetrs50_lion_dense10240.h5', '.h5'),
        ('https://storage.googleapis.com/csac_final_v2/new/styles_v9.csv', '.csv'),
        ('https://storage.googleapis.com/csac_final_v2/new/vgg16.h5', '.h5'),
        ('https://storage.googleapis.com/csac_final_v2/new/total.txt', '.txt'),
        ('https://storage.googleapis.com/csac_final_v2/new/12_final_v5(0806).csv', '.csv'),
        ("https://storage.googleapis.com/csac_final_v2/new/best_m.pt", '.pt'),
        ('https://storage.googleapis.com/csac_final_v2/new/style_mbti_v2.csv', '.csv')]

yolo_pt = None
style_csv = None
resnet_model = None
txt_db = None
color_csv = None
vgg16_model = None
mbti_data = None

for url, suffix in urls:
    file_name = url.split('/')[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    temp_file.write(chunk)
            if file_name == 'best_m.pt':
                yolo_pt = temp_file.name
            elif file_name == 'styles_v9.csv':
                style_csv = temp_file.name
            elif file_name == 'style_mbti_v2.csv':
                mbti_data = temp_file.name
            elif file_name == 'model_resnetrs50_lion_dense10240.h5':
                resnet_model = temp_file.name
            elif file_name == 'vgg16.h5':
                vgg16_model = temp_file.name
            elif file_name == 'total.txt':
                txt_db = temp_file.name
            elif file_name == '12_final_v5(0806).csv':
                color_csv = temp_file.name

st.set_page_config(page_title="KTA by ColdShower team", page_icon="random", layout="wide")

def load_image():
    image = "https://storage.googleapis.com/csac_final_v2/new/title.jpg"
    return image
st.image(load_image(), caption="", use_column_width=True)

with st.sidebar:
    selected = option_menu("Main Menu", ['Home', 'Know Thy Art','Neural Style Transfer','Artwork MBTI','Speech to Art to Speech', '미술박사'],
                       icons=['shop', 'palette','camera fill','puzzle','easel','lightbulb'], menu_icon="cast", default_index=0)

st.divider()

def load_and_resize_images():
    images = []
    for i in range(1, 6):
        img=f"https://storage.googleapis.com/csac_final_v2/new/home_{i}.jpg"
        images.append(img)
    return images

if selected == 'Home':
    st.header("Welcome to our Homepage!")
    images = load_and_resize_images()
    for img in images:
        st.image(img, use_column_width=True)


elif selected == 'Know Thy Art':
    with st.form(key="form"):
        source_img=st.file_uploader(label='Choose an image...', type=['png','jpg', 'jpeg'])
        submit_button = st.form_submit_button(label="Analyze")
        if submit_button:
            if source_img:
                uploaded_image = Image.open(source_img)
                if uploaded_image:
                    result = YOLO(yolo_pt).predict(uploaded_image)
                    result_plot = result[0].plot()[:, :, ::-1]               
                    with st.spinner("Running...."):
                        try:
                            result_2 = result[0]
                            box = result_2.boxes[0]                
                            cords = box.xyxy[0].tolist()
                            cords = [round(x) for x in cords]
                            area = tuple(cords)
                            @st.cache_data
                            def load_and_crop_image(source_img, area):
                                lc_img = uploaded_image.crop(area)
                                lc_img=copy.deepcopy(lc_img)
                                return lc_img
                            cropped_img = load_and_crop_image(source_img, area)
                            col1, col2,col3 = st.columns(3)
                            with col1:                               
                                st.image(image=uploaded_image,
                                         caption='Uploaded Image',
                                         use_column_width=True)    
                            with col2:
                                st.image(result_plot, caption="Detection Image", use_column_width=True)
                            with col3:
                                st.image(cropped_img, caption="Cropped Image", use_column_width=True)
                        except:
                            st.image(image=uploaded_image,
                                     caption='''Uploaded Image
                                     (Clean as whistle!)''',
                                     use_column_width=True)  
                            @st.cache_data
                            def uncropped_img():
                                uc_img=copy.deepcopy(uploaded_image)
                                return uc_img
                            cropped_img=uncropped_img()
                        m = load_model(resnet_model)
                        x = img_to_array(cropped_img)
                        x = tf.image.resize(x, [224, 224])
                        x = np.array([x])
                        x = preprocess_input(x)                
                        predict = m.predict(x)                       
                        class_indices = {0: 'Abstract Expressionism', 1: 'Baroque', 2: 'Cubism', 3: 'Impressionism',4 : 'Primitivism',5:'Rococo',6:'Surrealism'}  
                        korean_class_indices={0:'추상표현주의 (Abstract Expressionism)',
                                              1:'바로크 (Baroque)',
                                              2:'입체주의 (Cubism)',
                                              3:'인상주의 (Impressionism)',
                                              4:'원시주의 (Primitivism)',
                                              5:'로코코 (Rococo)',
                                              6:'초현실주의 (Surrealism)'}                    
                        top_3_indices = np.argsort(predict[0])[-3:][::-1]
                        top_3_labels = [class_indices[index] for index in top_3_indices]
                        top_3_probabilities = [predict[0][index] * 100 for index in top_3_indices]
                        top_prediction_index = np.argsort(predict[0])[-1]
                        st.divider()
                        col1,col2=st.columns(2)
                        with col1:
                            
                            st.markdown("<h2 style='text-align: center; color: black;'>Top 3 Predicted Classes</h2>", unsafe_allow_html=True)
                            fig, ax = plt.subplots()
                            wedges, texts= ax.pie(
                                top_3_probabilities, labels=['', '', ''], 
                                startangle=90, 
                                pctdistance=0.8, 
                                labeldistance=0.7,
                                colors=['#161953','#B3CEE5','#FAC898']
                                )
                            circle = plt.Circle((0, 0), 0.6, color='white')
                            ax.add_artist(circle)
                            top_3_info=[]
                            for index in top_3_indices:
                                class_label = class_indices[index]
                                probability = predict[0][index] * 100
                                top_3_info.append(f'{class_label} ({probability:.2f}%)')
                            ax.legend(wedges, top_3_info, loc='lower center', fontsize=12, bbox_to_anchor=(0, -0.2, 1, 1))
                            st.pyplot(fig)  
                        with col2:
                            st.title('')
                            st.write(st.secrets['clova_url'])
                            st.title('')
                            st.markdown("<h3 style='text-align: center; color: black;'>❣️ 해당 그림의 사조는<br></h3>", unsafe_allow_html=True)
                            st.markdown(f"<h2 style='text-align: center; color: #161953;'>{korean_class_indices[top_prediction_index]}<br></h2>", unsafe_allow_html=True)
                            st.markdown("<h3 style='text-align: center; color: black;'>와 가장 비슷합니다.<br></h3>", unsafe_allow_html=True)
                            
                            st.title('')
                            
                            sajo = korean_class_indices[top_prediction_index]                         
                            df = pd.read_csv(style_csv)
                            matching_rows = df[df['style'] == class_indices[top_prediction_index]]                                
                            matching_apps = matching_rows['app'].values
                            matching_exps = list(matching_rows['exp'].values)[0]
                            
                            col1, col2 = st.columns([1, 6])
                            if len(matching_apps) > 0:
                                for app in matching_apps:
                                    col2.markdown(app,unsafe_allow_html=True)                                
                                encText = urllib.parse.quote("해당 그림의 사조는" + sajo + "와 가장 비슷합니다."+matching_exps)
                                data = st.secrets['clova_data'] + encText;
                                request = urllib.request.Request(st.secrets['clova_url'])
                                st.write(st.secrets['clova_url'])
                                request.add_header("X-NCP-APIGW-API-KEY-ID",st.secrets['clova_id'])
                                request.add_header("X-NCP-APIGW-API-KEY",st.secrets['clova_secret'])
                                response = urllib.request.urlopen(request, data=data.encode('utf-8'))
                                rescode = response.getcode()
                                if(rescode==200):
                                    response_body = response.read()
                                    with tempfile.TemporaryFile(suffix=".mp3") as temp:
                                        temp.write(response_body)
                                        temp.seek(0)
                                        audio_file = open(temp.name, 'rb')
                                        audio_bytes = audio_file.read()
                                        base64_bytes = base64.b64encode(audio_bytes)
                                        base64_string = base64_bytes.decode()
                                        st.markdown(f'<audio autoplay controls><source src="data:audio/mp3;base64,{base64_string}"></audio>', unsafe_allow_html=True)                         
                            else:
                                st.subheader("No related apps found for the predicted art style.")
                                
                        try:
                            if not cropped_img.tobytes() == Image.new('RGB', (5, 5), color='white').tobytes():
                                st.divider()
                                st.subheader('')
                                st.markdown("<h2 style='text-align: center;'>Color Analysis</h2>", unsafe_allow_html=True)
                                st.subheader('')
                                cropped =cropped_img.convert('RGB')                            
                                image_bytes = io.BytesIO()
                                cropped.save(image_bytes, format='JPEG')
                                image_bytes.seek(0)
                                color_thief = ColorThief(image_bytes)
                                dominant_color = color_thief.get_color()
                                color_palette = color_thief.get_palette(color_count=6)
                                dominant_color_list = []
                                color_palette_list = []
                                dominant_color_list.append(dominant_color)
                                color_palette_list.append(color_palette)
                                def rgb_to_hex(rgb_tuple):
                                    r, g, b = rgb_tuple
                                    return "#{:02x}{:02x}{:02x}".format(r, g, b)            
                                code = rgb_to_hex(dominant_color)
                                col1,col2=st.columns(2)
                                with col1:
                                    st.image(cropped_img,use_column_width=True)                        
                                st.subheader('')
                                with col2:
                                    st.title('')
                                    col1,col2=st.columns(2)
                                    with col1:
                                        st.markdown("<p style='padding:1.5em;font-size:20px;'>Dominant Color : </p>", unsafe_allow_html=True)
                                    with col2:
                                        st.title('')
                                        st.color_picker(label='dominant_color',value=code,label_visibility='collapsed')
                                    st.title('')
                                    st.markdown("<p style='padding:1.5em;font-size:20px;'>Color Palette : </p>", unsafe_allow_html=True)
                                    st.title('')
                                    columns = st.columns(7)
                                    for i, color in enumerate(color_palette):
                                        hex_color = rgb_to_hex(color)
                                        columns[i+1].color_picker(label=f"Color {i+1}", value=hex_color,label_visibility='collapsed')
                                st.divider()
                                st.subheader('')
                                st.markdown("<h2 style='text-align: center; color: black;'>Artworks with similiar colors</h2>", unsafe_allow_html=True)
                                st.subheader('')
                                def rgb_distance(rgb1, rgb2): 
                                    r1, g1, b1 = rgb1
                                    r2, g2, b2 = rgb2
                                    return ((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2) ** 0.5
                                        
                                def find_closest_color(rgb_color, color_list):
                                    closest_color = None
                                    min_distance = float('inf')
                                    closest_color_index = None
                                        
                                    for i, color_name in enumerate(color_list):
                                        distance = rgb_distance(rgb_color, webcolors.name_to_rgb(color_name))
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_color = color_name
                                            closest_color_index = i
                                        
                                    return closest_color, closest_color_index
                                        
                                rgb_color = dominant_color_list[0]
                                color_names = ['orangered', 'bisque', 'sandybrown', 'linen', 'antiquewhite', 'lavender', 'darkslateblue', 
                                                       'lightsteelblue', 'steelblue', 'midnightblue', 'cadetblue', 'wheat', 'goldenrod', 'palegoldenrod', 
                                                       'beige', 'khaki', 'rosybrown', 'indianred', 'maroon', 'darkolivegreen', 'darkkhaki', 'darkseagreen', 
                                                       'olivedrab', 'tan', 'sienna', 'peru', 'saddlebrown', 'burlywood', 'darkslategray', 'thistle', 'dimgray', 
                                                       'silver', 'gray', 'darkgray', 'lightgray', 'gainsboro', 'lightslategray', 'slategray', 'whitesmoke', 'palevioletred', 'black']
                                        
                                closest_color, closest_color_index = find_closest_color(rgb_color, color_names)
                                simcol_df = pd.read_csv(color_csv)
                                selected_rows = simcol_df[simcol_df['rep_clr'] == closest_color]
                                group = selected_rows.iloc[0]['group']
                                selected_rows = simcol_df[simcol_df['web_cg_dt'] == group]
                                random_sample = selected_rows.sample(n=9)
                                file_names = random_sample['file_name'].tolist()
                            
                                folder_paths = ["https://storage.googleapis.com/csac_final_v2/new/abstract_expressionism_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/nap_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/symbolism_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/rc_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/cu_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/bq_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/northern_renaissance_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/impressionism_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/romanticism_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/sr_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/expressionism_img/",
                                                        "https://storage.googleapis.com/csac_final_v2/new/realism_img/"]
                                        
                                files = ['abstract_expressionism_', 'nap_', 'symbolism_', 'rc_', 'cu_', 'bq_', 'orthern_renaissance',
                                                      'impressionism_', 'romanticism_', 'sr_', 'expressionism_', 'realism_']
                                def get_style_filename(prefix, number):
                                    idx = files.index(prefix)
                                    folder_path = folder_paths[idx]
                                    filename = f'{prefix}{number}.jpg'
                                    file_path = folder_path+filename
                                    return file_path
                                numbers = file_names
                                plt.figure(figsize=(10, 10))
                                for i, num in enumerate(numbers):
                                    for prefix in files:
                                        if num.startswith(prefix):
                                            number = num[len(prefix):]
                                            file_path = get_style_filename(prefix, number)
                                            image = Image.open(file_path)
                                            plt.subplot(3, 3, i + 1)
                                            plt.imshow(image)
                                            plt.axis('off')
                                                    
                                            row = simcol_df[simcol_df['file_name'] == num]
                                            if not row.empty:
                                                title = row['Title'].values[0]
                                                painter = row['Painter'].values[0]
                                                plt.annotate(textwrap.fill(f"{title}", width=35), (0,0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
                                                n1 = (len(title)) // 35 
                                                if (len(title)) % 35 == 0:
                                                    n1 -= 1
                                                y1 = -23 - 13*n1
                                                plt.annotate(textwrap.fill(f"by {painter}", width=35), (0,0), (0, y1), xycoords='axes fraction', textcoords='offset points', va='top')
                                plt.tight_layout(h_pad=5)
                                st.pyplot(plt.gcf())
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.divider()
                                st.subheader('')            
                                st.markdown("<h2 style='text-align: center; color: black;'>Artworks with similiar styles</h2>", unsafe_allow_html=True)
                                m = load_model(vgg16_model)
                                x = img_to_array(cropped_img)
    
                                x = tf.image.resize(x, [224, 224])
                                x = np.array([x])
                                predict = m.predict(pinp(x))
                                
                                total=txt_db
                                index_predict = total['predict']                            
                                similarities = []                                        
                                for i in index_predict:
                                    similarities.append(cosine_similarity(predict, i))                                            
                                x = np.array(similarities).reshape(-1,)                                            
                                top_9 = total.iloc[np.argsort(x)[::-1][:9]].reset_index(drop=True)                                            
                                top_9['url'] = top_9['url'].apply(lambda x: 'gs://csac_final_v2/new/paintings/' + x)                                    
                                plt.figure(figsize=(10, 10))
                                i = 1                                    
                                for idx, url in enumerate(top_9['url']):
                                    image = Image.open(url)
                                    plt.subplot(3, 3, i)
                                    plt.imshow(image)
                                    plt.axis('off')
                                    i += 1
                                    title = top_9['title'][idx]
                                    painter = top_9['painter'][idx]
    
                                    plt.annotate(textwrap.fill(f"{title}", width=35), (0,0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
                                    n1 = (len(title)) // 35 
                                    if (len(title)) % 35 == 0:
                                        n1 -= 1
                                    y1 = -23 - 13*n1
                                    plt.annotate(textwrap.fill(f"by {painter}", width=35), (0,0), (0, y1), xycoords='axes fraction', textcoords='offset points', va='top')
                        
                                plt.tight_layout(h_pad = 5)
                                st.pyplot(plt.gcf())
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                        except:
                            st.subheader('')            
                else:
                    st.subheader('You didnt upload your image')
            else:
                st.write("Please upload an image")               

elif selected=='Neural Style Transfer':
    st.title('Neural Style Transfer')
    st.header('')
    col1, col2 = st.columns(2)
    with col1:
        original_image = st.file_uploader(label='Choose an original image', type=['jpg', 'jpeg'])
        if original_image : 
                st.image(image=original_image,
                         caption='Original Image',
                         use_column_width=True)
    with col2: 
        style_image = st.file_uploader(label='Choose a style image', type=['jpg', 'jpeg'])
        if style_image :
                st.image(image=style_image,
                         caption='Style Image',
                         use_column_width=True)    
    st.header('')
    button=None
    def load_image(image_file, image_size=(512, 256)):
        content = image_file.read()
        img = tf.io.decode_image(content, channels=3, dtype=tf.float32)[tf.newaxis, ...]
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img
    if original_image and style_image :
        col1,col2,col3,col4,col5 = st.columns(5)
        with col3 : 
            button = st.button('Stylize Image')
            if button :
                with st.spinner('Running...') :
                    
                    original_image = load_image(original_image)
                    style_image = load_image(style_image)
                    
                    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
                    @st.cache_resource
                    def ais():
                        ais_model=tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
                        return ais_model
                    stylize_model = ais()
            
                    results = stylize_model(tf.constant(original_image), tf.constant(style_image))
                    stylized_photo = results[0].numpy().squeeze()
                    stylized_photo_pil = PIL.Image.fromarray(np.uint8(stylized_photo * 255)) 
    st.header('')
    col1,col2,col3 = st.columns(3)
    if button :
        with col2 :
            st.image(image=stylized_photo_pil,
                             caption='Stylized Image',
                             use_column_width=True)
            rain(
                emoji="🎈",
                font_size=30,
                falling_speed=10,
                animation_length="infinite"
                )
elif selected=='Artwork MBTI':
    def resize_image(image, width, height):
        return image.resize((width, height), Image.Resampling.LANCZOS)
    
    def sequential_matchup_game(images, image_folder, mbti_data):
        st.subheader("더 마음에 드는 사진을 골라주세요 :smile:")
        st.write("본 미니게임은 11라운드로 진행되는 토너먼트식 게임입니다.")
        image_list = list(zip(range(len(images)), images))
        match_count = 0
        width, height = 750, 572
        while len(image_list) > 1:
            match_count += 1
            st.write(f"{match_count}번째 라운드 :point_down: ")
            col1, col2 = st.columns(2)
            image_1 = image_list[0]
            image_2 = image_list[1]
            with col1:
                st.image(resize_image(image_1[1], width, height), use_column_width=True, caption='첫번째 이미지')
            with col2:
                st.image(resize_image(image_2[1], width, height), use_column_width=True, caption='두번째 이미지')
    
            choice = st.radio(f"어느 쪽이 더 좋나요? {match_count} 번째 선택", ('선택안함', '첫번째 이미지', '두번째 이미지'))
    
            st.write('-----------')
    
            if choice == '선택안함':
                st.write("선택을 진행해주세요. 당신의 MBTI 유형을 맞혀보겠습니다. :bulb:")
                break
            
            elif choice == '첫번째 이미지':
                image_list.append(image_1)
                image_list.pop(0)
                image_list.pop(0)
                if match_count != 11:
                    st.info('선택을 마쳤습니다. 스크롤을 내려 다음 라운드를 진행해주세요.', icon="ℹ️")
                else:
                    st.info('모든 라운드가 끝났습니다. 스크롤을 내려 결과를 확인해주세요.', icon="ℹ️")
            
            elif choice =='두번째 이미지':
                image_list.append(image_2)
                image_list.pop(0)
                image_list.pop(0)
                if match_count != 11:
                    st.info('선택을 마쳤습니다. 스크롤을 내려 다음 라운드를 진행해주세요.', icon="ℹ️")
                else:
                    st.info('모든 라운드가 끝났습니다. 스크롤을 내려 결과를 확인해주세요.', icon="ℹ️")
                
            st.write('-----------')
            
    
        if len(image_list) == 1:
            winner_image = image_list[0]
            st.subheader("경기 종료!")
            st.write("최종 선택을 받은 작품은 :")
            st.image(resize_image(winner_image[1], width, height), use_column_width=True)
    
            mt = mbti_data.iloc[winner_image[0]]
            mbti_exp_info = mt['exp']
            mbti_short = mt['mbti']
            mbti_style = mt['style']
            st.subheader(mbti_style + " 작품이 제일 마음에 드는 당신의 MBTI 유형은....")
            st.subheader(mbti_short + ' 입니까:question:')
            st.write(mbti_exp_info)
    
    def main():
        st.title("Mini Game - 미술사조 mbti test :heart:")
        image_folder = "gs://csac_final_v2/new/mbti/"
        image_names = [f"img_{i}.jpg" for i in range(1, 13)]
        images = [Image.open(image_folder + name) for name in image_names]
        conn = st.experimental_connection('gcs', type=FilesConnection)
        mbti_data=pd.read_csv(mbti_data)
        sequential_matchup_game(images, image_folder, mbti_data)
    
    if __name__ == "__main__":
        main()

elif selected == 'Speech to Art to Speech':

    if 'load_state' not in st.session_state:
        st.session_state.load_state = False
        
    if 'generate_state' not in st.session_state:
        st.session_state.generate_state = False
    
    if 'load_state_2' not in st.session_state:
        st.session_state.load_state_2 = False
        
    if 'generate_state_2' not in st.session_state:
        st.session_state.generate_state_2 = False
    
    tab1, tab2 = st.tabs(["Impressionism", "Surrealism"])
    
    os.environ["REPLICATE_API_TOKEN"] = st.secrets['replicate_api_token']
    
    def speak(text):
         tts = gTTS(text=text, lang='ko', slow=False)
         with tempfile.TemporaryFile(suffix=".mp3") as temp:
            temp.write(tts)
            temp.seek(0)
            audio_file = open(temp.name, 'rb')
            audio_bytes = audio_file.read()
            base64_bytes = base64.b64encode(audio_bytes)
            base64_string = base64_bytes.decode()
            st.markdown(f'<audio autoplay controls><source src="data:audio/mp3;base64,{base64_string}"></audio>', unsafe_allow_html=True)
    
    @st.cache_resource
    def generate_image(img_description):
        output = replicate.run(
            st.secrets['img_generator_1'],
            input={"prompt": f"{img_description}"})
        return output
    
    @st.cache_resource
    def generate_image_2(img_description):
        output = replicate.run(
            st.secrets['img_generator_2'],
            input={"prompt": f"{img_description}"})
        return output
    
    @st.cache_resource
    def generate_text(img_description):
        output = replicate.run(
            st.secrets['multimodal_llm'],
            input={"prompt" : st.secrets['img_prompt'],
                "img":image})
        text=[]
        for item in output:
            text.append(item)
        text=''.join(text)
        return text
    
    @st.cache_resource
    def translate_ko(text):
        translator = deepl.Translator(st.secrets['deepl_api']) 
        result = translator.translate_text(text, target_lang='KO') 
        return result.text
    
    @st.cache_resource
    def translate_en(text):
        translator = deepl.Translator(st.secrets['deepl_api']) 
        result = translator.translate_text(text, target_lang='EN-US') 
        return result.text
    
    def naver_clover_tts(text) :
        
        client_id = st.secrets['clova_id_2']
        client_secret = st.secrets['clova_secrets_2']
        
        encText = urllib.parse.quote(text)
        data = st.secrets['clova_data'] + encText;
        url = st.secrets['clova_url']
        
        request = urllib.request.Request(url)
        request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
        request.add_header("X-NCP-APIGW-API-KEY",client_secret)
        response = urllib.request.urlopen(request, data=data.encode('utf-8'))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            with tempfile.TemporaryFile(suffix=".mp3") as temp:
                temp.write(response_body)
                temp.seek(0)
                audio_file = open(temp.name, 'rb')
                audio_bytes = audio_file.read()
                base64_bytes = base64.b64encode(audio_bytes)
                base64_string = base64_bytes.decode()
                st.markdown(f'<audio autoplay controls><source src="data:audio/mp3;base64,{base64_string}"></audio>', unsafe_allow_html=True)
    
    with tab1 :
    
        st.title("Speech to Art to Speech")
        st.subheader(':blue[Impressionism] :male-artist:')
    
        st.markdown("""---""")
        
        wav_audio_data = st_audiorec()
        
        if wav_audio_data or st.session_state.load_state:
            st.session_state.load_state = True
            try : 
                temp_audio_file = "temp_audio.wav"
                with open(temp_audio_file, "wb") as f:
                    f.write(wav_audio_data)
                
                speech_sr = SR.Recognizer()
                
                with SR.AudioFile(temp_audio_file) as source:
                    audio = speech_sr.record(source)
                    
                text = speech_sr.recognize_google(audio_data=audio, language='ko-KR')
                
                st.markdown("""---""")
                
                st.markdown("<h3 style='text-align: left; color: black;'> 입력된 음성 : <br></h3>", unsafe_allow_html=True)
                st.write(text)
                
                translated_text = translate_en(text)
                
                st.markdown("<h3 style='text-align: left; color: black;'> 한영 번역: <br></h3>", unsafe_allow_html=True)
                st.write(translated_text)
                
                st.markdown("""---""")
                
                os.remove(temp_audio_file)
                
                img_description = st.text_input(label='Image Description', value=translated_text)
                
                generate = st.button('Generate Impressionist Painting')
                            
                if generate or st.session_state.generate_state:
                    st.session_state.generate_state = True
                    st.markdown("""---""")
                    generated_img = generate_image(img_description)
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림 작품 : <br></h3>", unsafe_allow_html=True)
                    st.image(generated_img)
                    
                    image= generated_img[0]
                    
                    text = generate_text(image)
                    
                    st.markdown("""---""")
                    
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림에 대한 설명 : <br></h3>", unsafe_allow_html=True)
                    st.write(text)
                    
                    translated_text_2 = translate_ko(text)
                    
                    st.markdown("<h3 style='text-align: left; color: black;'> 영한 번역 : <br></h3>", unsafe_allow_html=True)
                    st.write(translated_text_2)
                    
                    st.write('')
                    
                    button = st.button(label='음성지원 :mega:') 
                    
                    st.markdown("""---""")
                    
                    if button :
                        naver_clover_tts(translated_text_2)
                        
                else:
                    pass
            except :
                st.write('다시 시도해 주시길 바랍니다.')
                
        else:
            pass
    
    with tab2 :
            
        st.title("Speech to Art to Speech")
        st.subheader(':red[Surrealism] :art:')
    
        st.markdown("""---""")
        
        wav_audio_data = st_audiorec()
        
        if wav_audio_data or st.session_state.load_state_2:
            st.session_state.load_state_2 = True
            try : 
                temp_audio_file = "temp_audio.wav"
                with open(temp_audio_file, "wb") as f:
                    f.write(wav_audio_data)
                
                speech_sr = SR.Recognizer()
                
                with SR.AudioFile(temp_audio_file) as source:
                    audio = speech_sr.record(source)
                    
                text = speech_sr.recognize_google(audio_data=audio, language='ko-KR')
                
                st.markdown("""---""")
                
                st.markdown("<h3 style='text-align: left; color: black;'> 입력된 음성 : <br></h3>", unsafe_allow_html=True)
                st.write(text)
                
                translated_text = translate_en(text)
                
                st.markdown("<h3 style='text-align: left; color: black;'> 한영 번역: <br></h3>", unsafe_allow_html=True)
                st.write(translated_text)
                
                st.markdown("""---""")
                
                os.remove(temp_audio_file)
                
                img_description = st.text_input(label='Image Description', value=translated_text)
                
                generate_2 = st.button('Generate Surrealist Painting')
                            
                if generate_2 or st.session_state.generate_state_2:
                    st.session_state.generate_state_2 = True
                    st.markdown("""---""")
                    generated_img = generate_image_2(img_description)
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림 작품 : <br></h3>", unsafe_allow_html=True)
                    st.image(generated_img)
                    
                    image= generated_img[0]
                    
                    text = generate_text(image)
                    
                    st.markdown("""---""")
                    
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림에 대한 설명 : <br></h3>", unsafe_allow_html=True)
                    st.write(text)
                    
                    translated_text_2 = translate_ko(text)
                    
                    st.markdown("<h3 style='text-align: left; color: black;'> 영한 번역 : <br></h3>", unsafe_allow_html=True)
                    st.write(translated_text_2)
                    
                    st.write('')
                    
                    button = st.button(label='음성지원 :loudspeaker:') 
                    
                    st.markdown("""---""")
                    
                    if button :
                        naver_clover_tts(translated_text_2)
                        
                else:
                    pass
            
            except :
                st.write('다시 시도해 주시길 바랍니다..')
                
        else:
            pass

elif selected == '미술박사':
    with st.form('chatbot'):
        OP=Options()
        OP.add_argument('--headless=new')
        OP.add_experimental_option(
            "prefs",
            {
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
                "profile.default_content_setting_values.notifications": 2           
            },
        )
        OP.add_argument('--disable-notifications') 
        OP.add_argument("--disable-infobars")
        OP.add_argument("--disable-extensions")
        OP.add_argument("--start-maximized");
        OP.add_argument("--window-size=1920,1080");
        OP.add_argument('--ignore-certificate-errors')
        OP.add_argument('--allow-running-insecure-content')
        OP.add_argument("--disable-web-security")
        OP.add_argument("--disable-site-isolation-trials")
        OP.add_argument("--user-data-dir=C:\\Users\\home\\Downloads")
        OP.add_argument("--disable-features=NetworkService,NetworkServiceInProcess")
        OP.add_argument("--test-type")
        OP.add_argument('--no-sandbox')
        OP.add_argument('--disable-gpu')
        OP.add_argument('--profile-directory=Default')
        OP.add_argument('--user-data-dir=C:/Temp/ChromeProfile')
        
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(**kwargs):
            return openai.Completion.create(**kwargs)
        
        openai.api_key = st.secrets['openai_api']
        
        uploaded_file = 'C:/streamlit/v2_new_art_df.csv'
        art_df = pd.read_csv(uploaded_file, encoding='utf-8')
        df_ref=art_df.Reference
        df=art_df.drop(columns=['Reference'])
        
        buffer=io.StringIO()
        df.info(buf=buffer)
        info_str=buffer.getvalue()
    
        @st.cache_data
        def art_info():
            conn = st.experimental_connection('gcs', type=FilesConnection)
            result=conn.read("gs://csac_final_v2/new/v2_new_art_df.csv", input_format='csv')
            return result
        art_df=art_info()
        df_ref=art_df.Reference
        df=art_df.drop(columns=['Reference'])
    
        buffer=io.StringIO()
        df.info(buf=buffer)
        info_str=buffer.getvalue()
        
        def return_answer(user_input):
            if user_input:
                try:
                    my_bar=st.progress(0, text='질문 내용 분석중...')
                    pre_prompt_1=f'''If the question below is about art, return 0, otherwise return 1.
        
                    "{user_input}" 
                    '''
                    response = completion_with_backoff(
                      model="text-davinci-003",
                      prompt=pre_prompt_1,
                      temperature=1,
                      max_tokens=10,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0
                    )
                    qna_c=int(response.choices[0].text.strip())
                    if qna_c == 0 : 
                        my_bar.progress(10, text='미술 쪽 질문이 맞는 것 같습니다.')
                        pre_prompt_2=f'''If there is a question related to any of the following: "Art movements, characteristics of art, artists in art, representative art works, and the historical context of art," please output 0; otherwise, output 1.
        
                        "{user_input}" 
                        '''
                        response_1 = completion_with_backoff(
                          model="text-davinci-003",
                          prompt=pre_prompt_2,
                          temperature=1,
                          max_tokens=10,
                          top_p=1,
                          frequency_penalty=0,
                          presence_penalty=0
                        )
                        qna_c_1=int(response_1.choices[0].text.strip())                        
                        if qna_c_1 == 0:
                            my_bar.progress(25, text='KTA팀의 Database를 사용하겠습니다.')
                            prompt_1 = f'''Look at this carefully, while thinking step by step.
                            
                            "{user_input}" 
                            
                            We'll call this a 'Input' from now on.
                            I need a sqlite3 query to retrieve the contents related to this 'Input' from a pandas.DataFrame named 'df'. Below are the df.info(), df.head() df.iloc[0].T of df that you should refer to when writing your query. Note that df.head() is created exactly as it is in df.
                            
                            {info_str}
                            
                            {df.head()}
                            
                            {df.iloc[0].T}
                            
                            I'll teach you some precautions when writing queries. You must follow these precautions when answering. Any answer that does not follow these precautions will greatly interfere with my work.
                            
                            1. write the query by interpreting or translating the language appropriately to make it easier to apply the above question to df.
                            If you need to rephrase the question in your query, be sure to refer to the texts described in df.head(), df.iloc[0].T, and df.info().
                            
                            2. Always write query statements using only the information in df.
                            
                            3. answer the query in the form of a "1-line sql query" starting with 'SELECT' and ending with ';'. Do not attach any text other than the requested answer. Do not include any '\n' or line breaks. Your answer should look like a one-line SQL query.
                            '''    
                            response_1 = completion_with_backoff(
                              model="text-davinci-003",
                              prompt=prompt_1,
                              temperature=1,
                              max_tokens=256,
                              top_p=1,
                              frequency_penalty=0,
                              presence_penalty=0
                            )    
                            criteria=re.search(r'SELECT.*?;',response_1.choices[0].text.strip())
                            if criteria:
                               response_2=' '.join(criteria.group().split())
                            ref_genre=[]
                            for i in list(df.사조):
                                if i in response_2:
                                    ref_genre.append(i)
                            my_bar.progress(50, text='데이터 조사 준비 완료')
                            result = ''
                            if response_2: 
                                conn = sqlite3.connect(':memory:')
                                df.to_sql('df', conn, index=False)
                                query = response_2
                                cursor = conn.cursor()
                                cursor.execute(query)
                                result_rows = cursor.fetchall()
                                columns = [desc[0] for desc in cursor.description]
                                result_df = pd.DataFrame(result_rows, columns=columns)
                                for i, row in result_df.iterrows():
                                    result += f"index_number: {i} \n"
                                    for column in result_df.columns:
                                        result += f"{column}: {row[column]}"
                                        if column != result_df.columns[-1]:
                                            result += " \n"
                                    result += " \n\n"
                                if result == '':
                                    raise ValueError()
                            my_bar.progress(75, text='데이터 조사 완료.')
                            prompt_2=f'''The following content is written as a single line of text for a dataframe:
                                
                                {result}
        
                    Using the information above, respond to the text below in Korean language, written in fluent and correct grammar.
                    
                    "{user_input}"'''
                            my_bar.progress(90, text='답변을 생각하는 중..')
                            response_3 = completion_with_backoff(
                              model="text-davinci-003",
                              prompt=prompt_2,
                              temperature=1,
                              max_tokens=900,
                              top_p=1,
                              frequency_penalty=0,
                              presence_penalty=0
                            )
                            final_response = response_3.choices[0].text.strip()
                            if ref_genre:
                                final_response=final_response+'\n\n\n[Reference]\n'+'\n'.join(art_df.loc[art_df['사조'].isin(ref_genre),'Reference'].tolist())
                            else:
                                final_response=final_response+'\n\n\n[Reference]\n'+'KTA팀 DB'
                            my_bar.progress(100, text='답변 완료.')
                        if qna_c_1 == 1:
                            my_bar.progress(25, text='Web Search 모드 실행.')
                            search_prompt = f'''Below is a question. Replace it with an Korean search term that the questioner would use to get the answer they want when searching on Google, and most important expression in your Korean search term should be enclosed in ".
                            
                            {user_input} 
                            '''
                            search_response = completion_with_backoff(
                              model="text-davinci-003",
                              prompt=search_prompt,
                              temperature=1,
                              max_tokens=256,
                              top_p=1,
                              frequency_penalty=0,
                              presence_penalty=0
                            )    
                            search_response_1=search_response.choices[0].text.strip()
                            my_bar.progress(30, text=f'"{search_response_1}" 검색중..')
                            driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=OP)
                            driver.get(f'https://www.google.com/search?q={search_response_1}')
                            try:
                                WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="result-stats"]')))
                            finally:
                                for i in range(1, 4):
                                    try:
                                        element = driver.find_element(By.XPATH, f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/div/span/a')
                                        element.send_keys(Keys.ENTER)
                                        break
                                    except:
                                        pass
                           
                            WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
                            my_bar.progress(50, text='조사 자료 분석중..')
                            html=driver.page_source
                            soup=BeautifulSoup(html,'html.parser')
                            del html
                            temp_soup=soup.select_one('body').text
                            my_bar.progress(85, text='최종 자료를 기반한 답변 작성중..')
                            response_test = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo-16k",
                      messages=[
                        {
                          "role": "system",
                          "content": "You are a helpful assistant."
                        },
                        {
                          "role": "user",
                          "content": f"""Find the answers in the resources I provide and answer my questions in Korean with correct grammar. Below are the questions and resources.
                          Question : {user_input}
                          
                          Resources : 
                          {temp_soup}"""
                        }
                      ],
                      temperature=1,
                      max_tokens=512,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0
                    )
                            search_resp_fin=response_test['choices'][0]['message']['content']
                            
                            if search_resp_fin:
                                final_response=search_resp_fin+'\n\n[Reference]\n\n'+f"{driver.title}\n\n"+f"{driver.current_url}"
                                driver.quit()
                                my_bar.progress(100, text='답변 완료.')
                            else :
                                final_response='좋은 답변으로 삼을만한 Reference를 찾지 못했습니다. 죄송합니다ㅠㅠ'
                                driver.quit()
                                my_bar.progress(100, text='답변 완료.')
                    if qna_c == 1:
                        my_bar.progress(100, text='분석 완료')
                        final_response='''미술과 관련된 궁금증이나 질문을 주신 것이 확실한가요?\n전 <b>요청사항의 형식</b>, <b>미술과 관련이 없는 내용</b>, 또는 <b>궁금증이나 질문이 아닌 것</b>들을 답변할 수 없습니다.'''
                except Exception:
                    my_bar.progress(50, text='좀더 확실한 답변을 위해 Web Search 모드가 실행됩니다.')
                    search_prompt = f'''Below is a question. Replace it with an Korean search term that the questioner would use to get the answer they want when searching on Google, and most important expression in your Korean search term should be enclosed in ".
                    
                    {user_input} 
                    '''
                    search_response = completion_with_backoff(
                      model="text-davinci-003",
                      prompt=search_prompt,
                      temperature=1,
                      max_tokens=256,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0
                    )    
                    search_response_1=search_response.choices[0].text.strip()
                    my_bar.progress(60, text=f'"{search_response_1}" 검색중..')
                    driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=OP)
                    driver.get(f'https://www.google.com/search?q={search_response_1}')
                    try:
                        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="result-stats"]')))
                    finally:
                        for i in range(1, 4):
                            try:
                                element = driver.find_element(By.XPATH, f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/div/span/a')
                                element.send_keys(Keys.ENTER)
                                break
                            except:
                                pass
                    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
                    my_bar.progress(70, text='조사한 자료 분석중..')
                    html=driver.page_source
                    soup=BeautifulSoup(html,'html.parser')
                    del html
                    temp_soup=soup.select_one('body').text
                    my_bar.progress(95, text='최종 자료를 기반한 답변 작성중..')
                    response_test = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-16k",
              messages=[
                {
                  "role": "system",
                  "content": "You are a helpful assistant."
                },
                {
                  "role": "user",
                  "content": f"""Find the answers in the resources I provide and answer my questions in Korean with correct grammar. Below are the questions and resources.
                  Question : {user_input}
                  
                  Resources : 
                  {temp_soup}"""
                }
              ],
              temperature=1,
              max_tokens=512,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
                    search_resp_fin=response_test['choices'][0]['message']['content']
                    if search_resp_fin:
                        final_response=search_resp_fin+'\n\n[Reference]\n\n'+f"{driver.title}\n\n"+f"{driver.current_url}"
                        driver.quit()
                        my_bar.progress(100, text='답변 완료.')
                    else :
                        final_response='좋은 답변으로 삼을만한 Reference를 찾지 못했습니다. 죄송합니다ㅠㅠ'
                        driver.quit()
                        my_bar.progress(100, text='답변 완료.')
            return st.markdown(f'<p>{final_response}</p>',unsafe_allow_html=True)
        st.title("더 궁금하신 것이 있으신가요?")
        col1,col2=st.columns([8.75,1.25])
        with col1:
            if 'key' not in st.session_state:
                st.session_state['key'] = 'value'
            your_question_input = st.text_input("미술과 관련된 모든 궁금증과 질문들을  작성하신 후 엔터를 눌러주세요!")
        with col2:
            st.subheader('')
            submitted = st.form_submit_button("Submit")
        if submitted:
            try:
                st.session_state['key'] = 'value_1'
                st.divider()
                return_answer(your_question_input)
            except:
                st.empty()
    try:
        if st.session_state['key'] == 'value_1':
            user_feedback=st.radio('맘에 드시는 기능인가요?',['맘에 드셨다면 네, 싫으셨다면 아니오를 골라주세요.','네','아니오'])
            if user_feedback=='네':
                thank_you=st.write('사용해주셔서 감사합니다💕💕💕 다른 질문을 또 해주세요!')
                time.sleep(3)
                for key in st.session_state.keys():
                    del st.session_state[key]
                if 'key' not in st.session_state:
                    st.session_state['key'] = 'value'
            elif user_feedback=="아니오":
                st.divider()
                st.empty()
                st.title('저희 팀에게 피드백을 주세요!')
                st.markdown("""
                **사용하시면서 불편하셨던 점에 대한 의견을 전부 적어서 저희들에게 알려주세요. 더 나은 서비스로 보답드릴 것을 약속드립니다 - KTA Team(`chohk4198@gmail.com`)**
                """)
                contact_form="""<form action="https://formsubmit.co/chohk4198@gmail.com" method="POST">
                     <label for="message">Feedback</label><br>
                     <textarea id="message" name="message" rows="10" cols="100" placeholder="여기에 메시지를 입력하세요."  required></textarea><br><br>
                     <label for="email">email</label><br>
                     <input type="email" name="email" size='81' required>
                     <button type="submit">Send</button>
                </form>"""
                st.markdown(contact_form,unsafe_allow_html=True)
            elif user_feedback=='맘에 드셨다면 네, 싫으셨다면 아니오를 골라주세요.': 
                st.session_state['key'] == 'value_1'
    except:
        st.empty()
