import numpy as np
from joblib import load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import pybase64
import sklearn
import tempfile
from time import sleep

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return pybase64.b64encode(image_file.read()).decode('utf-8') 
    


col1, col2 = st.columns(2)

with col1:

    job2 = Image.open('img/mestrado.png')
    st.image(job2, use_container_width=True)
    
with col2:

    html_page_title = """
<div style="background-color:black;padding=60px">
        <p style='text-align:center;font-size:60px;font-weight:bold; color:white'>Classificador de Candidatos</p>
</div>
"""               
    st.markdown(html_page_title, unsafe_allow_html=True)

logo = Image.open('img/logo.png')
st.sidebar.image(logo, use_container_width=True)

# Carregando o modelo Random Forest
def load_modelo():
    modelo = load("pipeline_rfc_mestrado.pkl")
    #st.success("Modelo carregado com sucesso.")
    return modelo

# Função para gerar gráfico de importância
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def center_img(image, size):
    # Centralizar imagem da cloud
    img = Image.open(f'img/{image}.png')
    image_path="image.png"
    img.save(image_path)
    # Getting the base64 string
    base64_image = encode_image(image_path)
    # Usando HTML para centralizar a imagem
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64_image}" alt="Imagem" style="width: {size}%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
        )


def show_feature_importance():
    clf = modelo.named_steps['classifier']
    
    
    #num_features2 = 'experiencia'
    feature_names = ['idade',	'experiencia_anos',	'qtd_artigos',	'nota_projeto_pesquisa',	'nota_curriculo', 'nivel_ingles']
    
    # Importâncias
    importances = clf.feature_importances_
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    st.table(df)
    # Gráfico
    #st.markdown("### Importância das Variáveis")
    #fig, ax = plt.subplots(figsize=(10, 15))
    #df['Feature'] = df['Feature'].astype(str)
    #ax.barh(df["Feature"], df["Importance"], color="orange")
    #ax.set_title("Ranking das Features - Random Forest")
    #ax.set_xlabel("Importância")
    #ax.set_ylabel("Feature")
    #plt.tight_layout()
    
    # Adicionar os valores na frente das barras
    #for i, (value, name) in enumerate(zip(df['Importance'], df['Feature'])):
    #    plt.text(value + 0.01, i, f"{value * 100:.2f}%", va='center')

    #st.pyplot(fig)
    
st.sidebar.markdown('# Métricas:')
st.sidebar.markdown('## Accuracy: 90%')
st.sidebar.markdown('## F1-score: 73% para classe 1')
st.sidebar.write('-----')
st.sidebar.markdown('### Observações: ')
st.sidebar.markdown('### Existem apenas 20 vagas disponíveis.')
st.sidebar.markdown('### Os primerios 40 serão considerados aprovados nessa fase inicial.')
st.sidebar.markdown('### Haverá uma entrevista e os 20 melhores serão admitidos no mestrado.')



st.write(' ') 
st.markdown("### Informe:")
st.write(' ') 
col01,col02, col03 = st.columns(3) 
with col01:
    st.markdown("#### Idade")
    idade = st.slider("Idade", 21, 40, 25, label_visibility='collapsed')  

    st.markdown("#### Nota Proj Pesq")
    nota_proj_pes = st.slider("Idade", 4, 10, 7, label_visibility='collapsed')    
    
with col02:
    st.markdown("#### Experiência")
    experiencia_anos = st.slider("Experiência?", 0, 10, 3, label_visibility='collapsed') 

    st.markdown("#### Nota Curriculo")
    nota_curriculo = st.slider("Curriculo?", 0, 10, 4, label_visibility='collapsed')    

with col03:
    st.markdown("#### Artigos")
    qtd_artigos = st.slider("Idade", 0, 10, 3, label_visibility='collapsed')  
     
    st.markdown("#### Nível Inglês")
    nivel_ingles = st.slider("Ingles", 0, 4, 3, label_visibility='collapsed')      



dados = [idade,  nota_proj_pes, experiencia_anos, nota_curriculo, qtd_artigos, nivel_ingles]
colunas = ['idade',  'nota_projeto_pesquisa', 'experiencia_anos', 'nota_curriculo', 'qtd_artigos', 'nivel_ingles']

features = pd.DataFrame([dados], columns=colunas)

st.markdown("### Perfil do candidato")
st.table(features)

modelo = load_modelo()

st.sidebar.markdown("### Modelo:")
st.sidebar.write(str(modelo[-1][1]))
#st.write("Modelo:", modelo[1].__class__.__name__)

if st.button("Classificar"):

    # Previsão do modelo
    pred = modelo.predict(features)
    probs = modelo.predict_proba(features)

    col1, col2, col3 = st.columns([0.3, 0.2, 0.6])
    
    with col1:
        # Exibe previsão e probabilidades
        st.markdown("### Previsão")
    
        classe = 'Aprovado' if pred == 1 else 'Não Aprovado'
    
        st.write(classe.upper())
        
        colunas_prob = ['Não Aprovado', 'Aprovado']  # Ajustar a ordem se necessário
        df_probs = pd.DataFrame(probs, columns=colunas_prob)
    
        st.markdown("### Probabilidades")
        st.dataframe(df_probs.T.style.format("{:.2%}"))
    
    with col2:    
        colunas_prob = ['Não Aprovado', 'Aprovado']  # Ajustar a ordem se necessário
        df_probs = pd.DataFrame(probs, columns=colunas_prob)

      
    
    with col3:
        # Gerar resultados (com emojis e formatação HTML)
        resultado = f"""
    <div style='font-size: 50px; font-weight: bold; text-align: center;'>
        {'😄 Aprovado 😊 ' if classe == 'Aprovado' else '☹️ Reprovado'}
    </div>
    """
        st.markdown(resultado, unsafe_allow_html=True)
    
        if classe == 'Não Aprovado':
            st.markdown("#### Tópicos mais importantes na avaliação.")
            show_feature_importance()
        else:
            html_page_subtitle = """
          <div style="background-color:black;padding=60px">
            <p style='text-align:center;font-size:30px;font-weight:bold; color:white'></p>
          </div>
        """               
            st.markdown(html_page_subtitle, unsafe_allow_html=True)
            # Imagem da comemoracao
            img = 'chop'        
            center_img(img, 100)
            st.balloons()
            sleep(5)        
            st.balloons()        
        
    


