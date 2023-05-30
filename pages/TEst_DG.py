
import streamlit as st
import pandas as pd
import functions.sfunctions as sf


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