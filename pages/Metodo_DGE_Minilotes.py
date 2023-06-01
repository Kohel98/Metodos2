import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as stc 






tab1, tab2 = st.tabs(["Definiciones","Ejemplo y Aplicacion"])
with tab1:
    
        
    st.title(":red[**Descenso de gradiente de mini lotes**]")
    st.header("Teorema")
    """
        En Machine Lerning, el descenso de gradiente es una técnica de optimización 
        utilizada para calcular los parámetros del modelo (coeficientes y sesgo) para algoritmos 
        como la regresión lineal, la regresión logística, las redes neuronales, etc. 
        
        En esta técnica, iteramos repetidamente a través del conjunto de 
        entrenamiento y actualizamos el modelo. parámetros de acuerdo con el 
        gradiente del error con respecto al conjunto de entrenamiento. 
        
        :red[Descenso de gradiente de minilote:] los parámetros se actualizan después de calcular 
        el gradiente del error con respecto a un subconjunto del conjunto de entrenamiento.
        
        Dado que se considera un subconjunto de ejemplos de entrenamiento, 
        puede realizar actualizaciones rápidas en los parámetros del modelo 
        y también puede aprovechar la velocidad asociada con la vectorización del código.
        
        Según el tamaño del lote, las actualizaciones se pueden hacer menos ruidosas: cuanto mayor 
        sea el tamaño del lote, menos ruidosa será la actualización.
        
        Por lo tanto, el descenso de gradiente de mini lotes hace un compromiso entre la convergencia 
        rápida y el ruido asociado con la actualización de gradiente, lo que lo convierte en un algoritmo
        más flexible y robusto.
    """
with tab2:
    st.title(":blue[Mini Batch Gradient Descent en Python]")   
    st.text("Librerias que se necesitan")
    code = '''
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import time
        from scipy import stats 
        import sklearn
        '''
    st.code(code, language='python')
    st.text("Emulamos y Proyectamos los datos")
    code = '''
        %matplotlib inline
            def timeit(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                print('%r  %2.2f ms' % \
                    (method.__name__, (te - ts) * 1000))
            return result
        return timed
        
       X, y = sklearn.datasets.make_regression(n_samples = 10000, 
                       n_features=1, 
                       n_informative=1, 
                       noise=20,
                       random_state=2019)
        plt.scatter(X,y)
            
           x = X.flatten()
slope, intercept,_,_,_ = stats.linregress(x,y)
print (slope)
print (intercept) 
        
        '''
    st.code(code, language='python')
    st.text("Obtenemos el intercepto y la pendiente")
    code = '''
       
slope, intercept,_,_,_ = stats.linregress(x,y)
print (slope)
print (intercept) 
        
        '''
    st.code(code, language='python')
    st.text("Graficamos la aproximacion con lamda")
    code = '''
       
y = y.reshape(-1,1) 
        
        '''
    st.code(code, language='python')
    st.text("Definimos las funciones")
    code = """
        @timeit
    def minibatch_gradient_descent(X,y,theta,learning_rate=0.01, iterations=10, batch_size =20):
        m = len(y)
        cost_history = np.zeros(iterations)
        n_batches = int(m/batch_size)
        for it in range(iterations):
            cost =0.0
            indices = np.random.permutation(m)
            X = X[indices]
            y = y[indices]
            for i in range(0,m,batch_size):
                X_i = X[i:i+batch_size]
                y_i = y[i:i+batch_size]
                X_i = np.c_[np.ones(len(X_i)),X_i]
                prediction = np.dot(X_i,theta)
                theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
                cost += cal_cost(theta,X_i,y_i)
            cost_history[it]  = cost
        return theta, cost_history
    
    """
    st.code(code, language='python')
    code = """
    lr = 0.05
    n_iter = 1000
    theta = np.random.randn(2, 1)
    theta,cost_history = minibatch_gradient_descent(X,y,theta,lr,n_iter)
    print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
    print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
    
    """
    st.code(code, language='python')
   
  