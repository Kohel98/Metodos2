import streamlit as st
import pandas as pd
import numpy as np

tab1, tab2 = st.tabs(["Definiciones","Ejemplo y Aplicacion"])
with tab1:
    st.title(":blue[**Descenso del gradiente estocastico**]")
    st.header("Teorema")
    """
    La palabra ' estocástico ' significa un sistema o proceso vinculado con una probabilidad aleatoria. Por lo tanto, en Stochastic Gradient Descent, se
    seleccionan aleatoriamente algunas muestras en lugar de todo el conjunto de datos para cada iteración. En Gradient Descent, hay un término llamado "lote"
    que denota el número total de muestras de un conjunto de datos que se utiliza para calcular el gradiente para cada iteración. En la optimización típica de Gradient Descent, como Batch Gradient Descent, el lote se toma como el
    conjunto de datos completo. Aunque usar todo el conjunto de datos es realmente útil para llegar a los mínimos de una manera menos ruidosa y menos aleatoria, el problema surge cuando nuestro conjunto de datos crece. 
    
     En SGD, utiliza solo una sola muestra, es decir, un tamaño de lote de uno, para realizar cada iteración. 
     La muestra se mezcla aleatoriamente y se selecciona para realizar la iteración.
     
     Stochastic Gradient Descent (SGD) es una variante del algoritmo Gradient Descent que se utiliza 
     para optimizar los modelos de aprendizaje automático. En esta variante, solo se usa un ejemplo de entrenamiento aleatorio para }
     calcular el gradiente y actualizar los parámetros en cada iteración.
     
     Desventajas:
      + :blue[Actualizaciones ruidosas:] Las actualizaciones en SGD son ruidosas y tienen una varianza alta,
      lo que puede hacer que el proceso de optimización sea menos estable y generar oscilaciones alrededor del mínimo.
      
      + :blue[Convergencia lenta:] SGD puede requerir más iteraciones para converger al mínimo, 
      ya que actualiza los parámetros para cada ejemplo de entrenamiento uno a la vez.
      
      + :blue[Convergencia lenta:] SGD puede requerir más iteraciones para converger al mínimo, 
      ya que actualiza los parámetros para cada ejemplo de entrenamiento uno a la vez.
      
      + :blue[Menos preciso:] debido a las actualizaciones ruidosas, SGD puede no converger al mínimo global exacto y puede resultar en una solución subóptima.
      Esto se puede mitigar mediante el uso de técnicas como la programación de la tasa de aprendizaje y las actualizaciones basadas en el impulso.
      
      SGD es generalmente más ruidoso que el descenso de gradiente típico, generalmente se necesita una mayor cantidad de iteraciones para alcanzar los mínimos, 
      debido a la aleatoriedad en su descenso. A pesar de que requiere una mayor cantidad de iteraciones para alcanzar los mínimos que el descenso de gradiente típico, 
      todavía es computacionalmente mucho menos costoso que el descenso de gradiente típico. Por lo tanto, en la mayoría de los escenarios, se prefiere SGD a Batch Gradient Descent 
      para optimizar un algoritmo de aprendizaje.
    
    """
with tab2:
  st.title(":blue[Stochastic Gradient Descent]")
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
st.text("Definimos nuestra funciones")
code = """
@timeit
def stocashtic_gradient_descent(X, y, theta, learning_rate=0.01, iterations=10):
    m = len(y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        cost = 0.0
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = X[rand_ind, :].reshape(1, X.shape[1])
            y_i = y[rand_ind].reshape(1, 1)
            prediction = np.dot(X_i, theta)
            theta = theta - (1 / m) * learning_rate * (X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta, X_i, y_i)
        cost_history[it] = cost
    return theta, cost_history
"""
st.code(code, language='python')
code = """
  lr = 0.05
  n_iter = 1000
  theta = np.random.randn(2, 1)
  X_b = np.c_[np.ones((len(X),1)),X]
  theta,cost_history = stocashtic_gradient_descent(X_b,y,theta,lr,n_iter)
  print("Theta0: {:0.3f},\nTheta1:{:0.3f}".format(theta[0][0], theta[1][0]))
  print("Final cost/MSE:  {:0.3f}".format(cost_history[-1]))
"""
st.code(code, language='python')
code = """
  fig, ax = plt.subplots(figsize=(10, 8))
  ax.set_ylabel("{J(Theta)}", rotation=0)
  ax.set_xlabel("{Iterations}")
  theta = np.random.randn(2, 1)
  _ = ax.plot(range(n_iter), cost_history, "b.")

"""
st.code(code, language='python')
    