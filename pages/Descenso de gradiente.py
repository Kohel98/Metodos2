
import streamlit as st
import pandas as pd
import functions.sfunctions as sf


tab1, tab2 = st.tabs(["Definiciones","Aplicaciones"])


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
    # pylint: disable=line-too-long
    def write():
        #"""Used to write the page in the app.py file"""
        with st.spinner("Cargando Página de inicio ..."):
            #ast.shared.components.title_awesome(" - Homepage")
            #st.title("LiRA Web App - HomePage")
            
            st.write('''
                    Descenso de la gradiente 
                    el proposito es simular un descenso de una gradiente 
                    
        
            ####
            ## 1. Simular la funcion lineal
            
            Normalmente, durante la regresión lineal, intentamos predecir una función lineal del tipo
            Y = aX + b a partir de las muestras de datos. En este caso no hay conjunto de datos, por lo que vamos a crear una predefinida y configurable
            predefinida y configurable para generar estos datos. Para ello necesitamos especificar lo siguiente:
            
            #### Distribución de x (Input)
            Para generar muestras de datos aleatorios para la entrada X, necesitaremos conocer la Media y la Desviación de la distribución,
            que pueden establecerse ajustando los controles respectivos en el widget de la barra lateral.
            
            #### Coeficientes (a, b)
            La Pendiente y el Intercepto de la función lineal también pueden ajustarse utilizando los controles disponibles en el widget de la barra lateral.
            
            ####
            ## 2. Generar dato aleatorio de poblacion
        Una vez que hemos simulado una función lineal, necesitamos infundir estos datos con algo de ruido, para permitir que el algoritmo descubra
            la función lineal simulada. Para ello debemos especificar el número de muestras "n" y la media de la distribución del error
            o Media del Error (Residual):
                
            #### Numero de Muestras - n
            Para generar los datos, es necesario especificar el número de puntos de datos según el control correspondiente de la barra lateral.
            
            #### Residual - e
        Distribución del error que se añadirá a Y para generar las Muestras_Y.      
            ''')
            
            # ****************************************************************************************************
            # Input widgets
            # ****************************************************************************************************
            
            st.sidebar.title("Controles de la aplicación")
            st.sidebar.subheader('**Numero de muestras**')
            n = st.sidebar.slider('Seleccionar el número de muestras para la población',5, 100)
            
            st.sidebar.subheader('Configurar la distribución para X')
            # Use these for good visual - (mean_x = 3, stddev_x = 2)
            mean_x = st.sidebar.slider("Seleccione la media para generar X",-10,10,3)
            stddev_x = st.sidebar.slider('Select Standard Deviation for generating X',-5,5,2)
            
            st.sidebar.subheader('Coeficienstes')
            # Select a = 0.35 and b = 2.5 for good visual
            a = st.sidebar.slider('Seleccione "Pendiente" para la línea de regresión', -2.0, 2.0,0.15)
            b = st.sidebar.slider('Seleccione "Intercepción" para la línea de regresióne', -10.0, 10.0, 2.5)
            
            st.sidebar.subheader('Error residual')
            #st.write('Select residual distribution for noise added to Simulated Linear function')
            mean_res = st.sidebar.slider ('Seleccionar la media del error residual',0.0,2.0,0.7)
                            
            # Dataframe to store generated data
            rl, X1, y1 = sf.generate_data(n, a, b, mean_x, stddev_x, mean_res)
            
            st.write('''
            ##
            ## 3. Ver una muestra de los datos generados
            La siguiente tabla muestra una muestra de la población generada "X" e "y" junto con "Y_act", la salida real de la función lineal simulada utilizada para generar "y" observada. 
            función lineal simulada utilizada para generar la "y" observada.''')
        
            st.dataframe(rl.head())
            # ****************************************************************************************************************************
        
        
            st.write('''
            ##
            ## 4. Seleccione el método de regresión lineal
            Para aplicar el modelo de regresión lineal, hay 4 opciones disponibles:
            
            * Mínimos cuadrados ordinarios - Regresión lineal simple
            * Mínimos cuadrados ordinarios - Ecuaciones normales
            * Algoritmo de Descenso Gradiente o LSM
            * SKlearn - Modelos lineales
            
            Para profundizar en estos métodos, consulte las páginas de Recursos.
            #### Seleccione el método de regresión lineal
                    ''')
            
            method=["Descenso del Gradiente", "MCO-Regresión lineal simple", "Ecuaciones OLS-Normal"]
            lira_method = st.selectbox('',(method))
            #if st.button('Predict'):
            
            if lira_method == "Descenso del Gradiente":
                # Configuration Parameters for Gradient Descent
                L = st.slider('Seleccione la tasa de aprendizaje', 0.0,0.05, 0.015,0.0001)
                epochs = st.slider('Seleccione el número de iteraciones (Epochs)', 100, 1000,250,5)
                #pmethod = ['Altair','Plotly','Matplotlib', 'Skip Animation']
                #mode = st.selectbox("Select Plotting Library",(pmethod))
                
                # Calculate model and coefficients
                alpha, beta, ypred, tmp_err= sf.GD_method(rl, L, epochs)
                error = pd.DataFrame(tmp_err)
                            
                # Evaluate Model
                model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
                
                # Results summary
                # Plot final graphs and Evaluate Model metrics
                sf.GD_plots_and_metrics(rl, ypred, error, lira_method,model_coeff, model_assess)
                
                
            
            if lira_method == "MCO-Regresión lineal simple":
                # Calculate coefficients
                alpha, beta = sf.OLS_method(rl)
                
                # Calculate Regression Line
                ypred = sf.liraweb_predict(alpha, beta, rl['X'], lira_method)
                
                # Evaluate Model
                model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
            
                # Results summary
                # Plot final graphs and Evaluate Model metrics
                sf.plots_and_metrics(rl, ypred, lira_method,model_coeff, model_assess)
                
                
            if lira_method == "Ecuaciones OLS-Normal":
                # Calculate coefficients
                alpha, beta = sf.NE_method(X1,y1)
                
                # Calculate Regression Line
                ypred = sf.liraweb_predict(alpha, beta, rl['X'], lira_method)
                
                # Evaluate Model
                # create new evaluation method sf.NE_evaluation and replace - for now use ypred
                model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
                
                # Results summary
                # Plot final graphs and Evaluate Model metrics
                sf.plots_and_metrics(rl, ypred, lira_method,model_coeff, model_assess)
                
            
            if lira_method == 'SKlearn':
                # Import library
                from sklearn.linear_model import LinearRegression
                
                # Calculate model and coefficients
                lr = LinearRegression()
                lr.fit(rl['X'].values.reshape(-1, 1), rl['y'].values.reshape(-1, 1))
                alpha = lr.coef_[0][0]
                beta = lr.intercept_[0]
                ypred = sf.liraweb_predict(alpha, beta, rl['X'], lira_method)
                
                # Evaluate Model
                model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
                
                # Results summary
                # Plot final graphs and Evaluate Model metrics
                sf.plots_and_metrics(rl, ypred, lira_method,model_coeff, model_assess)
                

    if __name__ == "__main__":
        write()