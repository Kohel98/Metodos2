import streamlit as st
import pandas as pd
import numpy as np

tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplos","Aplicaciones"])
with tab1:
     st.title(":blue[Descenso de la Gradiente]")
     """
         Descenso de gradiente es conocido como uno de los algoritmos de optimización más utilizados
         para entrenar modelos de aprendizaje automático mediante la minimización de errores 
         entre los resultados reales y esperados.
         
         Además, el descenso de gradiente también se usa para entrenar redes neuronales.
         
         El :red[**objetivo principal**] del descenso de gradiente es minimizar la función convexa 
         mediante la iteración de actualizaciones de parámetros. 
         
         Una vez que se optimizan estos modelos de aprendizaje automático,
         estos modelos se pueden usar como herramientas poderosas para la inteligencia artificial 
         y varias aplicaciones informáticas.
         
         Este se define como uno de los algoritmos de optimización iterativos de aprendizaje automático 
         más utilizados para entrenar los modelos de aprendizaje automático y aprendizaje profundo.
         Ayuda a encontrar el mínimo local de una función.
         La mejor manera de definir el mínimo local o el máximo local de una función mediante el descenso de gradiente es la siguiente: 
         + Si nos movemos hacia un gradiente negativo o nos alejamos del gradiente de la función en el punto actual, dará el :red[**mínimo local**] de esa función.
         + Siempre que nos desplacemos hacia un gradiente positivo o hacia el gradiente de la función en el punto actual, obtendremos el :red[**máximo local**] de esa función.
            
         Su procedimiemto consiste en  usar un algoritmo de descenso de gradiente 
         en minimizar la función de costo usando la iteración. 
         Para lograr este objetivo, realiza dos pasos iterativamente:
         + Calcula la derivada de primer orden de la función para calcular el gradiente o pendiente de esa función.
         + Alejarse de la dirección del gradiente, lo que significa que la pendiente aumentó desde el punto actual en alfa veces, donde Alfa se define como Tasa de aprendizaje. 
         
         Es un parámetro de ajuste en el proceso de optimización que ayuda a decidir el lapso del tiempo de los pasos. Un ejemplo seria cuando se grafica una gradiente negativa (pasando las coordenadas del nuevo punto) con un proceso iterativo hasta encontrar el minimo local.
         Estos pasos se denomina como **taza de aprendizaje**. 
         
         Con una tasa de aprendizaje alta podemos cubrir más terreno en cada paso, pero corremos el riesgo de 
         sobrepasar el punto más bajo ya que la pendiente de la colina cambia constantemente.
         Con una tasa de aprendizaje muy baja, podemos movernos con confianza en la dirección del gradiente 
         negativo ya que lo estamos recalculando con tanta frecuencia. Una tasa de aprendizaje baja es más precisa, 
         pero calcular el gradiente requiere mucho tiempo, por lo que nos llevará mucho tiempo llegar al fondo.
                 
        :red[**Función de costo**] 
        Con esta por medio de la función de pérdida determinamos "Qué tan bueno" es nuestro modelo para hacer 
        predicciones para un conjunto dado de parámetros. La función de costo tiene su proria curva y sus gradientes.
        La pendiente de esta curvatura nos mostrara cómo actualizar nuestros parámetros para que el modelo sea más preciso.
                
         """
         
with tab2:
    """
    Mucho texto
    """
         
         
with tab3:
    def form_callback(data1, data2):    
        with open('notes.csv', 'a+') as f:    #Append & read mode
            f.write(f"{data1},{data2}\n")
    with st.form(key="my_form",clear_on_submit=True):
        st.write("Enter Note")
        stock_ticker_input = st.text_input('Stock', key='ticker')
        note_input = st.text_input('Note', key='note')
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write("Note", note_input, "stock_ticker", stock_ticker_input)
        form_callback(stock_ticker_input,note_input)
        
    st.dataframe(pd.read_csv("notes.csv",names=["Stock","Note"]),height=300)

    
    