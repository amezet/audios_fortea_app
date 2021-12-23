# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 09:13:08 2021

@author: amemd
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import s3fs

import json
import os
import re

import pickle

from PIL import Image

import nltk
from nltk.corpus import stopwords


from wordcloud import WordCloud

import spacy

nltk.download('stopwords')

os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.AWS_SECRET_ACCESS_KEY

fs = s3fs.S3FileSystem(anon=False)


st.set_page_config(
      page_title="Base Datos Audios Padre Fortea",
      layout="wide",
      initial_sidebar_state="expanded",
  )


directorio_mp3 = 'audios-fortea/mp3/'
directorio_pkl = './data_pkl/'
#directorio_json = './json_all/'
directorio_csv = './data_csv/'

@st.cache
def lee_ficheros():
    # carga listado completo de ficheros
    list_files_all = pd.read_csv(directorio_csv + 'Listado_archivos_todos.csv', sep=';')
    
    list_files_all['date'] = list_files_all['date'].apply(lambda x:pd.to_datetime(x, format='%d/%m/%Y'))
    
    # add columna con nombre sin tipo
    list_files_all['file_name'] = list_files_all.file.str.replace('.mp3', '', regex=True).replace('.MP3', '', regex=True)
    
    # carga diccionario especializado
    df_dicc = pd.read_csv(directorio_csv + 'diccionario_especializado_con_tildes.csv', sep=';', encoding='UTF-8')
    
    # carga libros Biblia
    libros_Biblia = pd.read_csv(directorio_csv + 'Libros_Biblia.csv', sep=';')
    
    # carga list_json
    list_json = list(pd.read_csv(directorio_csv + 'list_json.csv', sep=';')['json'].values)
    
    return(list_files_all, df_dicc, libros_Biblia, list_json)


list_files_all, df_dicc, libros_Biblia, list_json = lee_ficheros()


# lista todos los json en el directorio
#list_json = os.listdir(directorio_json)
#list_json = list(pd.read_csv(directorio_csv + 'list_json.csv', sep=';')['json'].values)
# lista con los nombres solos
list_json_name = [f.replace('.json', '') for f in list_json]



df_json = pd.DataFrame({'ini_name':list_json})
df_json['end_name'] = ''
# identificacion de titulos de ficheros
for i in range(0, len(df_json['ini_name'])):
    a = df_json['ini_name'][i].replace('.json', '')
    a = a.replace('P. Fortea', '')
    b = re.sub(pattern="[-]",
               repl="",
               string=a)
    c = re.sub(pattern="[\d]+",
               repl="",
               string=b)
    c = c.replace('  ', '')
    d = re.sub(pattern="^ ",
               repl="",
               string=c)
    e = re.sub(pattern="^, ",
               repl="",
               string=d)
    f = re.sub(pattern=" ª parte",
               repl="",
               string=e)
    
    if len(f) > 1:
        df_json['end_name'][i] = f
    else:
        df_json['end_name'][i] = df_json['ini_name'][i].replace('.json','')



# selecciona indices de los ficheros
indexes_json_files = [i for i in list_files_all.index if list_files_all.file_name.iloc[i] in list_json_name]

# filtra dataset resumen con los json que hay transcritos
df = list_files_all.loc[indexes_json_files, :].copy()




stopwords = nltk.corpus.stopwords.words('spanish')

# Para el POS Tagging

# jar = 'stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
# model = 'stanford-postagger-full-2020-11-17/models/spanish-ud.tagger'

# java_path = "C:/Program Files (x86)/Java/jre1.8.0_311/bin/java.exe"
# os.environ['JAVAHOME'] = java_path



def tokenizar_limpiar(texto_join):
    return([t for t in nltk.word_tokenize(texto_join) if t not in ['.', ',', '?']])

def has_numbers(input):
    return any(char.isdigit() for char in input)

def find_token_index(list_token_index, string):
    return([t for t in list_token_index if t[0] == string])

def spans(txt):
    #tokens=toktok.tokenize(txt)
    tokens = [t for t in txt.split()]
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)


@st.cache
def cargar_listados():

    with open(directorio_pkl + 'list_df_kw.pkl', 'rb') as f:
        list_df_kw = pickle.load(f)
    
    with open(directorio_pkl + 'list_texto_j.pkl', 'rb') as f:
        list_texto_j = pickle.load(f)
        
    with open(directorio_pkl + 'list_dic_sim.pkl', 'rb') as f:
        list_dic_sim = pickle.load(f)    
    
    with open(directorio_pkl + 'list_df_libros_Biblia.pkl', 'rb') as f:
        list_df_libros_Biblia = pickle.load(f)
        
    with open(directorio_pkl + 'list_resumen.pkl', 'rb') as f:
        list_resumen = pickle.load(f)
        
    return(list_df_kw, list_texto_j, list_dic_sim, list_df_libros_Biblia, list_resumen)


list_df_kw, list_texto_j, list_dic_sim, list_df_libros_Biblia, list_resumen = cargar_listados()



df['libros_Biblia'] =''
for i in range(len(list_df_libros_Biblia)):
    df_libros_Biblia = list_df_libros_Biblia[i]
    df_libros_Biblia = (df_libros_Biblia.loc[df_libros_Biblia.n_termino>0,:].copy().reset_index())
    
    df.reset_index(inplace=True)
    
    df.loc[i, 'libros_Biblia'] = str(list(zip(df_libros_Biblia['index'], df_libros_Biblia['n_termino'])))
    #df.loc[i, 'libros_Biblia'] = str(list(zip(df_libros_Biblia['index'], df_libros_Biblia['n_termino'])))
    
    df.set_index('index', inplace=True)
    

df['KeyWords'] =''
for i in range(len(list_df_kw)):
    df_list_df_kw = list_df_kw[i]
    df_list_df_kw = df_list_df_kw.loc[0:10, :].copy()
    
    df.reset_index(inplace=True)
    
    df.loc[i, 'KeyWords'] = str(list(zip(df_list_df_kw['Palabras'], df_list_df_kw['Repeticiones'])))
    
    df.set_index('index', inplace=True)    

df['year'] = df.date.apply(lambda x: x.year)

min_year = int(df.year.min())
max_year = int(df.year.max())


    
col_to_show = ['date', 'size', 'file', 'duration_min', 'libros_Biblia', 'KeyWords', 'year']

all_libros = list(set(libros_Biblia['Libro_sin'].apply(lambda x: x.lower()).values))
all_libros.sort()


all_Keywords = []
for df_kw_i in list_df_kw:
    all_Keywords.append(df_kw_i.loc[0:10, 'Palabras'].values)
all_Keywords = list(set([item for sublist in all_Keywords for item in sublist]))
all_Keywords.sort()



st.header('Base de Datos de Audios (Padre Fortea)')
st.write('---------------------')

logo = Image.open('logo_Fortea.jpg')
st.sidebar.image(logo)

st.sidebar.write('--------------')
st.sidebar.subheader('Filtros')
filtrado = st.sidebar.checkbox('Aplicar filtro')


#st.sidebar.subheader('Filtrado por libros Biblia')
opciones_libros_biblia = st.sidebar.multiselect('Filtrado por libros Biblia', all_libros)
opciones_libros_biblia_Y_O = st.sidebar.checkbox('Y (marcado) -- O (desmarcado)', True)


opciones_KeyWords = st.sidebar.multiselect('Filtrado por KeyWords', all_Keywords)
opciones_KeyWords_Y_O = st.sidebar.checkbox(' Y (marcado) -- O (desmarcado)', True)

with st.container():
    col1, col2, col3 = st.columns([10, 1, 10])
    
    
    with col1:
        opcion_duration = st.slider('Filtrado Duración (minutos)', int(np.floor(df.duration_min.min())), int(np.floor(df.duration_min.max())),
                                    (int(np.floor(df.duration_min.min())), int(np.floor(df.duration_min.max()))))
    with col2:
        st.empty()
    
    with col3:
        opcion_year = st.slider('Filtrado Años', min_year, max_year, (min_year, max_year))





df_to_show = df.loc[:, col_to_show].copy()
df_to_show.sort_values(by='date', inplace=True)



if filtrado == False:
    
    st.sidebar.write('---------------------')
    add_selectbox = st.sidebar.selectbox(
        'Elegir Audio', df_to_show['file'])

    with st.container():
        
        st.markdown('## <font color="red">Tabla</font>', unsafe_allow_html=True)
        st.write('Número de registros: ', df_to_show.shape[0], '/', df.shape[0],
                 '   Horas audio totales seleccionadas: ', np.round(df_to_show.duration_min.sum()/60, 1))
        st.dataframe( (df_to_show.style.format({'date': "{:%Y/%m/%d}"})
                       .set_properties(**{
                           'font-size': '10pt',
                           })), height=700)
        
        with col1:
            counts, bins = np.histogram((df_to_show.duration_min))
            st.bar_chart(pd.DataFrame(index=bins.astype(int)[:-1], data={'counts':counts}), 100, 150)
        
        with col3:
            st.bar_chart(df_to_show.year.value_counts(sort=False), 100, 150)
    
    st.write('---------------------')
    st.markdown('## <font color="red">Inspección Texto</font>', unsafe_allow_html=True)
    
    
    
        
    df_ = df.copy().reset_index(drop=True)
    indice = df_.loc[df_.file_name == add_selectbox[:-4],:].index[0]
    
    #st.write(indice)


    
else:
    if (len(opciones_libros_biblia)==0) & (len(opciones_KeyWords)==0):
        mask = np.array([True] * df.shape[0])
        
    elif(len(opciones_libros_biblia)>0) & (len(opciones_KeyWords)==0):
        
        if opciones_libros_biblia_Y_O==True:
            mask = np.array([True] * df.shape[0])
            for o in opciones_libros_biblia:
                mask = mask & df_to_show['libros_Biblia'].str.contains(o).values
        else:
            mask = np.array([False] * df.shape[0])
            for o in opciones_libros_biblia:
                mask = mask | df_to_show['libros_Biblia'].str.contains(o).values
    
    elif(len(opciones_libros_biblia)==0) & (len(opciones_KeyWords)>0):
        
        if opciones_KeyWords_Y_O==True:
            mask = np.array([True] * df.shape[0])
            for o in opciones_KeyWords:
                mask = mask & df_to_show['KeyWords'].str.contains(o).values
        else:
            mask = np.array([False] * df.shape[0])
            for o in opciones_KeyWords:
                mask = mask | df_to_show['KeyWords'].str.contains(o).values
    
    else:
        
        if opciones_KeyWords_Y_O==True:
            mask1 = np.array([True] * df.shape[0])
            for o in opciones_KeyWords:
                mask1 = mask1 & df_to_show['KeyWords'].str.contains(o).values
        else:
            mask1 = np.array([False] * df.shape[0])
            for o in opciones_KeyWords:
                mask1 = mask1 | df_to_show['KeyWords'].str.contains(o).values
                
        if opciones_libros_biblia_Y_O==True:
            mask2 = np.array([True] * df.shape[0])
            for o in opciones_libros_biblia:
                mask2 = mask2 & df_to_show['libros_Biblia'].str.contains(o).values
        else:
            mask2 = np.array([False] * df.shape[0])
            for o in opciones_libros_biblia:
                mask2 = mask2 | df_to_show['libros_Biblia'].str.contains(o).values
                
        mask = mask1 & mask2
    
    mask3 = (df_to_show.duration_min >= opcion_duration[0]) & (df_to_show.duration_min <= opcion_duration[1])
    mask = mask * mask3
    
    mask4 = (df_to_show.date.apply(lambda x: x.year) >= opcion_year[0]) & (df_to_show.date.apply(lambda x: x.year) <= opcion_year[1])
    mask = mask * mask4
    
    
    with st.container():
        
        st.markdown('## <font color="red">Tabla</font>', unsafe_allow_html=True)
        st.write('Número de registros: ', df_to_show[mask].shape[0], '/', df.shape[0],
                 '   Horas audio totales seleccionadas: ', np.round(df_to_show[mask].duration_min.sum()/60, 1))
        st.dataframe( (df_to_show[mask].style.format({'date': "{:%Y/%m/%d}"})
                       .set_properties(**{
                           'font-size': '10pt',
                           })), height=700)
            
        with col1:
            counts, bins = np.histogram((df_to_show[mask].duration_min))
            st.bar_chart(pd.DataFrame(index=bins.astype(int)[:-1], data={'counts':counts}), 100, 150)
        
        with col3:
            st.bar_chart(df_to_show[mask].year.value_counts(sort=False), 100, 150)
    
    st.write('---------------------')
    st.markdown('## <font color="red">Inspección Texto</font>', unsafe_allow_html=True)

    st.sidebar.write('---------------------')
    add_selectbox = st.sidebar.selectbox(
        'Elegir Audio', df_to_show[mask]['file'])
    
    df_ = df.copy().reset_index(drop=True)
    indice = df_.loc[df_.file_name == add_selectbox[:-4],:].index[0]
    
    #st.write(indice)


@st.cache
def read_file_audio(filename):
    with fs.open(filename, 'rb') as f:
        return f.read()

# @st.cache
# def get_audio(add_selectbox):
#     audio_file = open(directorio_wav + add_selectbox[:-4] + '.wav', 'rb')
#     audio_bytes = audio_file.read()
#     return(audio_bytes)


ver_audio = st.sidebar.checkbox('Ver Audio', False)

if ver_audio:
    audio_bytes = read_file_audio(directorio_mp3 + add_selectbox)
    st.sidebar.audio(audio_bytes, format='audio/ogg')
    
inspeccionar = st.checkbox('Inspeccionar Texto', False)


if inspeccionar:

    with open(directorio_json + add_selectbox[:-4] + '.json') as f:
        diccionario = json.loads(f.read())

    
    df_kw = list_df_kw[indice].copy()
    texto_j = list_texto_j[indice]
    max_rep = max(df_kw.Repeticiones)
    df_libros_Biblia = list_df_libros_Biblia[indice].copy()
    
    resumen = list_resumen[indice]
    resumen = resumen.replace('\n', '<br>')
    
    resumen_con_negrita = []
    for w in resumen.split(" "):
        # a,b = 'áéíóúÁÉÍÓÚ','aeiouAEIOU'
        # trans = str.maketrans(a,b)

        # w_sin_tilde = w.translate(trans)
        w_sin_tilde = w
        w_sin_tilde_sin_signos = w_sin_tilde.replace('.', '').replace(',', '').replace(';', '').replace('?', '')
        w_sin_tilde_sin_signos = w_sin_tilde_sin_signos.lower()
        
        if (w_sin_tilde_sin_signos in df_kw.iloc[0:10,0].values) & (w_sin_tilde_sin_signos in df_libros_Biblia.loc[df_libros_Biblia.n_termino!=0,:].index.values):
            w_negrita = '**<font color="red">' + w + '</font>**'
        elif (not(w_sin_tilde_sin_signos in df_kw.iloc[0:10,0].values)) & (w_sin_tilde_sin_signos in df_libros_Biblia.loc[df_libros_Biblia.n_termino!=0,:].index.values):
            w_negrita = "**<font color='red'>" + w + "</font>**"
        elif (w_sin_tilde_sin_signos in df_kw.iloc[0:10,0].values) & (not(w_sin_tilde_sin_signos in df_libros_Biblia.loc[df_libros_Biblia.n_termino!=0,:].index.values)):
            w_negrita = '**' + w + '**'
        else:
            w_negrita = w
        resumen_con_negrita.append(w_negrita)

    resumen_con_negrita = " ".join(resumen_con_negrita)
    
    
    list_token_index = []
    for token in spans(texto_j):
        list_token_index.append(token)
    len(list_token_index)
    
    texto_inicial = " ".join(diccionario['DisplayText'])
    
    texto_con_negrita = []
    for w in texto_inicial.split(" "):
        # a,b = 'áéíóúÁÉÍÓÚ','aeiouAEIOU'
        # trans = str.maketrans(a,b)
    
        # w_sin_tilde = w.translate(trans)
        w_sin_tilde = w
        w_sin_tilde_sin_signos = w_sin_tilde.replace('.', '').replace(',', '').replace(';', '').replace('?', '')
        w_sin_tilde_sin_signos = w_sin_tilde_sin_signos.lower()
        
        if (w_sin_tilde_sin_signos in df_kw.iloc[0:10,0].values) & (w_sin_tilde_sin_signos in df_libros_Biblia.loc[df_libros_Biblia.n_termino!=0,:].index.values):
            w_negrita = '**<font color="red">' + w + '</font>**'
        elif (not(w_sin_tilde_sin_signos in df_kw.iloc[0:10,0].values)) & (w_sin_tilde_sin_signos in df_libros_Biblia.loc[df_libros_Biblia.n_termino!=0,:].index.values):
            w_negrita = "**<font color='red'>" + w + "</font>**"
        elif (w_sin_tilde_sin_signos in df_kw.iloc[0:10,0].values) & (not(w_sin_tilde_sin_signos in df_libros_Biblia.loc[df_libros_Biblia.n_termino!=0,:].index.values)):
            w_negrita = '**' + w + '**'
        else:
            w_negrita = w
        texto_con_negrita.append(w_negrita)
    
    texto_con_negrita = " ".join(texto_con_negrita)
    
    col_1, col_2 = st.columns([7, 3])
        
    with col_1:
        
        #st.write('Nombre: ', add_selectbox)
        st.write('**Orden (id)**: ', list_files_all.loc[list_files_all.file_name==add_selectbox[:-4], 'id'].values[0],
                 '-- **Fecha**: ', list_files_all.loc[list_files_all.file_name==add_selectbox[:-4], 'date'].dt.strftime('%d/%m/%Y').values[0],
                 '-- **Número de palabras**: ',len(diccionario['Words']),
                 '-- **Duración (minutos)**: ', np.round(float(diccionario['OffsetDuration'][-1][0])/60, 2))
        # st.write('Fecha: ', list_files_all.loc[list_files_all.file_name==add_selectbox[:-4], 'date'].dt.strftime('%d/%m/%Y').values[0])
        # st.write('Número de palabras: ',len(diccionario['Words']))
        # st.write('Duración (minutos): ', np.round(float(diccionario['OffsetDuration'][-1][0])/60, 2))
        
        st.subheader('Texto')
        st.markdown(texto_con_negrita, unsafe_allow_html=True)
    
    
    tuples = list(zip(df_kw.Palabras, df_kw.Repeticiones_mayorado))
    wc = WordCloud(stopwords=stopwords, background_color="white", colormap="Dark2", max_font_size=100, collocations=False)
    wordcloud = wc.generate_from_frequencies(dict(tuples))
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    with col_2:
        st.subheader('WordCloud')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()
        
        st.write('---------------------')
        st.subheader('Palabras más repetidas')
        st.dataframe(df_kw)
        st.subheader('Menciones libros Biblia')
        #df_libros_Biblia['index_termino'] = df_libros_Biblia['index_termino'].astype(str)
        st.dataframe(df_libros_Biblia.loc[df_libros_Biblia.n_termino!=0, ['n_termino']], height=500)
        
        st.subheader('Colocaciones')
        t = [w.lower().replace('.', '').replace(',', '').replace(';', '').replace('?', '') for w in tokenizar_limpiar(texto_inicial)]
        colo = nltk.Text(t).collocation_list(10, 2)
        st.dataframe([c[0]+' '+c[1] for c in colo], height=400)
        
        st.subheader('Resumen (experimental)')
        st.markdown(resumen_con_negrita, unsafe_allow_html=True)
        
        
