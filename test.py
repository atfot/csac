# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:32:24 2023

@author: ITWILL
"""

import copy
import streamlit as st
st.set_page_config(page_title="KTA by ColdShower team", page_icon="random", layout="wide")
from st_files_connection import FilesConnection
from streamlit_option_menu import option_menu
import PIL
from PIL import Image
from colorthief import ColorThief
import webcolors
import io
import torch
import tempfile
from ultralytics import YOLO
from tensorflow.keras.applications.resnet_rs import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import gcsfs
import pandas as pd
import random
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
#from google.oauth2 import service_account
#from google.cloud import storage


def load_image():
    #image = conn.read('csac_final_v1/final_v1/streamlit_files/title_img.png', input_format='png')
    htp7='https://storage.googleapis.com/csac_final_v1/final_v1/streamlit_files/title_img.png'
    return htp7
st.image(load_image(), caption="", use_column_width=True)

with st.sidebar:
    selected = option_menu("Main Menu", ['Home', 'Know Thy Art','Neural Style Transfer','Artwork MBTI'], 
        icons=['shop', 'palette','camera fill','puzzle'], menu_icon="cast", default_index=0)
    
def load_and_resize_images():
    images = []
    for i in range(1, 6):
        conn = st.experimental_connection('gcs', type=FilesConnection)
        image = conn.read(f'csac_final_v1/final_v1/streamlit_files/home_{i}.jpg', input_format='jpg')
        images.append(image)
    return images
if selected == 'Home':
    st.header("Welcome to the Home page!")
    images = load_and_resize_images()
    for img in images:
        st.image(img, use_column_width=True)
        
elif selected == 'Know Thy Art':
    @st.cache_resource
    def yolo():
        conn = st.experimental_connection('git-lfs', type=FilesConnection)
        model = conn.read('best_m.pt',input_format='pt')
        result = torch.load(model)
        return result
    model = yolo()
    with st.form(key="form"):
        source_img=st.file_uploader(label='Choose an image...', type=['png','jpg', 'jpeg'])
        submit_button = st.form_submit_button(label="Analyze")
        if submit_button:
            if source_img:
                uploaded_image = Image.open(source_img)
                if uploaded_image:
                    result = model.predict(uploaded_image)
                    result_plot = result[0].plot()[:, :, ::-1]               
                    with st.spinner("Running...."):
                        try:
                            result_2 = result[0]
                            box = result_2.boxes[0]                
                            cords = box.xyxy[0].tolist()
                            cords = [round(x) for x in cords]
                            area = tuple(cords)
                            
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
                            
                            def uncropped_img():
                                uc_img=copy.deepcopy(uploaded_image)
                                return uc_img
                            cropped_img=uncropped_img()
                        @st.cache_resource
                        def rnrs50():
                            conn = st.experimental_connection('git-lfs', type=FilesConnection)
                            model = conn.read('model_resnetrs50_lion_dense10240.h5',input_format='hdf5')
                            result = load_model(model)
                            return result
                        m = rnrs50()
                        x = img_to_array(cropped_img)
                        x = tf.image.resize(x, [224, 224])
                        x = np.array([x])
                        x = preprocess_input(x)                
                        predict = m.predict(x)                       
                        class_indices = {0: 'Abstract Expressionism', 1: 'Baroque', 2: 'Cubism', 3: 'Impressionism',4 : 'Primitivism',5:'Rococo',6:'Surrealism'}  # Replace with the correct class indices and labels.
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
                            st.title('')
                            st.markdown("<h3 style='text-align: center; color: black;'>❣️ 해당 그림의 사조는<br></h3>", unsafe_allow_html=True)
                            st.markdown(f"<h2 style='text-align: center; color: #161953;'>{korean_class_indices[top_prediction_index]}<br></h2>", unsafe_allow_html=True)
                            st.markdown("<h3 style='text-align: center; color: black;'>와 가장 비슷합니다.<br></h3>", unsafe_allow_html=True)
                            st.title('')
                            
                            def styles_v4():
                                conn = st.experimental_connection('gcs', type=FilesConnection)
                                styles_df = conn.read("csac_final_v1/final_v1/streamlit_files/styles_v8.csv", input_format="csv")
                                return styles_df
                            df = styles_v4()
                            matching_rows = df[df['style'] == class_indices[top_prediction_index]]                                
                            matching_apps = matching_rows['app'].values
                            col1, col2 = st.columns([1, 6])
                            if len(matching_apps) > 0:
                                for app in matching_apps:
                                    col2.markdown(app,unsafe_allow_html=True)
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
                                def rgb_distance(rgb1, rgb2): # 유클리드 거리 공식
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
                                
                                def final_v5():
                                    conn = st.experimental_connection('gcs', type=FilesConnection)
                                    df = conn.read("csac_final_v1/final_v1/streamlit_files/12_final_v5(0806).csv", input_format="csv")
                                    return df
                                simcol_df = final_v5()
                                selected_rows = simcol_df[simcol_df['rep_clr'] == closest_color]
                                group = selected_rows.iloc[0]['group']
                                selected_rows = simcol_df[simcol_df['web_cg_dt'] == group]
                                random_sample = selected_rows.sample(n=9)
                                file_names = random_sample['file_name'].tolist()
                                # Assuming you have the necessary GCS setup and authentication in place
                                # You may need to install and import the appropriate GCS library, like google-cloud-storage
                                
                                # Replace folder_paths with the corresponding GCS paths
                                folder_paths = [
                                    "csac_final_v1/final_v1/streamlit_files/abstract_expressionism_img",
                                    "csac_final_v1/final_v1/streamlit_files/nap_img",
                                    "csac_final_v1/final_v1/streamlit_files/symbolism_img",
                                    "csac_final_v1/final_v1/streamlit_files/rc_img",
                                    "csac_final_v1/final_v1/streamlit_files/cu_img",
                                    "csac_final_v1/final_v1/streamlit_files/bq_img",
                                    "csac_final_v1/final_v1/streamlit_files/northern_renaissance_img",
                                    "csac_final_v1/final_v1/streamlit_files/impressionism_img",
                                    "csac_final_v1/final_v1/streamlit_files/romanticism_img",
                                    "csac_final_v1/final_v1/streamlit_files/sr_img",
                                    "csac_final_v1/final_v1/streamlit_files/expressionism_img",
                                    "csac_final_v1/final_v1/streamlit_files/realism_img",
                                ]
                                
                                files = [
                                    'abstract_expressionism_',
                                    'nap_',
                                    'symbolism_',
                                    'rc_',
                                    'cu_',
                                    'bq_',
                                    'northern_renaissance',
                                    'impressionism_',
                                    'romanticism_',
                                    'sr_',
                                    'expressionism_',
                                    'realism_',
                                ]
                                
                                numbers = file_names
                                plt.figure(figsize=(10, 10))
                                for i, num in enumerate(numbers):
                                    for prefix in files:
                                        if num.startswith(prefix):
                                            number = num[len(prefix):]
                                            @st.cache_data
                                            def get_style_filename(prefix, number):
                                                idx = files.index(prefix)
                                                folder_path = folder_paths[idx]
                                                filename = f'{prefix}{number}.jpg'
                                                file_path = f'{folder_path}/{filename}'
                                                return file_path
                                            file_path = get_style_filename(prefix, number)
                                            @st.cache_data
                                            def simcol_image(file_path):
                                                conn = st.experimental_connection('gcs', type=FilesConnection)
                                                image = conn.read(file_path)
                                                result=imread(image)
                                                return result
                                            image=simcol_image(file_path)
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
                                @st.cache_resource
                                def vgg_model():
                                    conn = st.experimental_connection('git-lfs', type=FilesConnection)
                                    model = conn.read('vgg16.h5',input_format='hdf5')
                                    result = load_model(model)
                                    return result
                                m = vgg_model()
                                x = img_to_array(cropped_img)
                                #x.shape
                                x = tf.image.resize(x, [224, 224])
                                x = np.array([x])
                                predict = m.predict(pinp(x))
                                
                                def total_db(): 
                                    conn = st.experimental_connection('gcs', type=FilesConnection) 
                                    total_df = conn.read('csac_final_v1/final_v1/streamlit_files/total.txt', input_format='pickle') 
                                    return total_df
                                total=total_db()
                                index_predict = total['predict']                            
                                similarities = []                                        
                                for i in index_predict:
                                    similarities.append(cosine_similarity(predict, i))                                            
                                x = np.array(similarities).reshape(-1,)                                            
                                # 가장 유사한 이미지 9개
                                top_9 = total.iloc[np.argsort(x)[::-1][:9]].reset_index(drop=True)                                            
                                top_9['url'] = top_9['url'].apply(lambda x: 'csac_final_v1/final_v1/streamlit_files3/paintings' + x)                                    
                                plt.figure(figsize=(10, 10))
                                i = 1                                    
                                for idx, url in enumerate(top_9['url']):
                                    
                                    def top_9_image(url):
                                        conn = st.experimental_connection('gcs', type=FilesConnection)
                                        image = conn.read(url, input_format='jpg')
                                        return image
                                    image = top_9_image(url)
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
    if original_image and style_image :
        col1,col2,col3,col4,col5 = st.columns(5)
        with col3 : 
            button = st.button('Stylize Image')
            if button :
                with st.spinner('Running...') :
                    @st.cache_data
                    def nst_image(image_file, image_size=(512, 256)):
                        content = image_file.read()
                        img = tf.io.decode_image(content, channels=3, dtype=tf.float32)[tf.newaxis, ...]
                        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
                        return img
                    original_image = nst_image(original_image)
                    style_image = nst_image(style_image)
                    
                    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
                    @st.cache_resource
                    def ais():
                        ais_model=tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
                        return ais_model
                    stylize_model = ais()
            
                    results = stylize_model(tf.constant(original_image), tf.constant(style_image))
                    stylized_photo = results[0].numpy().squeeze()  # Convert to NumPy array and squeeze dimensions
                    stylized_photo_pil = PIL.Image.fromarray(np.uint8(stylized_photo * 255))  # Convert to PIL image and rescale to [0, 255]
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
    
    
    @st.cache_data
    def main():
        st.title("Mini Game - 미술사조 mbti test :heart:")
        image_folder = "csac_final_v1/final_v1/streamlit_files/new_folder/"  # GCS 이미지 폴더 경로    
        image_names = [f"img_{i}.jpg" for i in range(1, 13)]  # 이미지 파일명 리스트
        
        def game_image(image_path):
            conn = st.experimental_connection('gcs', type=FilesConnection)
            image = conn.read(image_path, input_format='jpg')
            return image
        images = [game_image(image_folder + name) for name in image_names]
        conn = st.experimental_connection('gcs', type=FilesConnection)
        mbti_data = conn.read("csac_final_v1/final_v1/streamlit_files/style_mbti_v2.csv", input_format="csv")
        
        sequential_matchup_game(images, image_folder, mbti_data)
    
    if __name__ == "__main__":
        main()
