import streamlit as st
import pandas as pd
import numpy as np

tab1, tab2= st.tabs(["Definiciones","Aplicacion y Ejemplo"])
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
with tab2:

        st.title(":blue[Descenso de la Gradiente por lotes en Python]")   
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
        st.text("Definimos nuestras funciones")
        code = """
        
        def cal_cost(theta,X,y):
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/2*m) * np.sum(np.square(predictions-y))
        return cost
        
        @timeit
        def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
            m = len(y)
            cost_history = np.zeros(iterations)
         theta_history = np.zeros((iterations, 2))
         for it in range(iterations):
         prediction = np.dot(X, theta)
         theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
         theta_history[it, :] = theta.T
         cost_history[it] = cal_cost(theta, X, y)
            return theta, cost_history, theta_history
        """
        st.code(code, language='python')
        st.text("Empecemos con 1000 iteraciones y una tasa de aprendizaje de 0,05. Empezar con theta de una distribución gaussiana")
        code = """
            def cal_cost(theta,X,y):
            m = len(y)
            predictions = X.dot(theta)
            cost = (1/2*m) * np.sum(np.square(predictions-y))
            return cost
            
            @timeit
            def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
                m = len(y)
                cost_history = np.zeros(iterations)
            theta_history = np.zeros((iterations, 2))
            for it in range(iterations):
            prediction = np.dot(X, theta)
            theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
            theta_history[it, :] = theta.T
            cost_history[it] = cal_cost(theta, X, y)
                return theta, cost_history, theta_history
        """
        st.code(code, language='python')
        st.text("Vamos a trazar el historial de costes a lo largo de las iteraciones")
        code = """
                fig,ax = plt.subplots(figsize=(12,8))
                ax.set_ylabel('J(Theta)')
                ax.set_xlabel('Iterations')
                _=ax.plot(range(n_iter),cost_history,'b.')
            """
        st.code(code, language='python')
        st.text("Después de unas 60 iteraciones el coste es plano, por lo que las iteraciones restantes no son necesarias o no darán lugar a ninguna optimización adicional. Acerquémonos hasta la iteración 100 y veamos la curva")
        code = """
                fig,ax = plt.subplots(figsize=(10,8))
                _=ax.plot(range(100),cost_history[:100],'b.')
            """
        st.code(code, language='python')