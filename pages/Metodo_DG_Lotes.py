import streamlit as st
import pandas as pd
import numpy as np

tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplos","Aplicaciones"])
with tab1:
    st.title("Descenso de gradiente de lotes")
    st.header("Teorema")
    """
    Gradient Descent es un algoritmo de optimización que le ayuda a encontrar los pesos óptimos para su modelo. Lo hace probando varios pesos y encontrando los pesos que se ajustan mejor a los modelos, es decir, minimiza la función de costo. La función de costo se puede definir como la diferencia entre la producción real y la producción prevista. Por lo tanto, cuanto más pequeña es la función de costo, más cerca está el resultado previsto de su modelo del resultado real. La función de costo se puede definir matemáticamente como: 
    
    """
    st.latex(r""" y = \beta + \theta_{n} x_n """ )
    """
    
    Mientras que por otro lado, la tasa de aprendizaje del descenso del gradiente se representa como $alpha$
    """
    st.latex(r"""\alpha""")
    """La tasa de aprendizaje es el tamaño del paso dado por cada gradiente. Si bien una tasa de aprendizaje grande puede darnos valores mal optimizados para
    """
    st.latex(r"""\beta y \theta""") 
    """, la tasa de aprendizaje también puede ser demasiado pequeña, lo que requiere un incremento sustancial en el número de iteraciones necesarias para obtener el punto de convergencia (el punto de valor óptimo para beta y theta) . Este algoritmo nos da el valor de alpha , beta y theta como salida.
    . Para implementar un algoritmo de descenso de gradiente necesitamos seguir 4 pasos:
    
    +Inicializar aleatoriamente el sesgo y el peso theta
    +Calcular el valor predicho de y que es Y dado el sesgo y el peso
    +Calcular la función de costo a partir de los valores pronosticados y reales de Y
    +Calcular pendiente y los pesos.
    
    Se inicia tomando un valor aleatorio para el sesgo y las ponderaciones, que en realidad podría estar cerca del sesgo y las ponderaciones óptimos o puede estar lejos.
    """
  