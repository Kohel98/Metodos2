import numpy as np
import pandas as pd
import random
import pylab
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sy


def compute_cost_function(m, t0, t1, x, y):
  return 1/2/m * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=1500):
    converged = False
    iter = 0
    m = x.shape[0] # numero de muestras

    # theta iniical
    t0 = 0
    t1 = 0

    # total error, J(theta)
    J = compute_cost_function(m, t0, t1, x, y)
    st.write('J=', J);
    # ciclo iterativo
    num_iter = 0
    while not converged:
        # para cada muestra de entrenamiento, calcular el gradiente (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])

        # Actualiza theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
    
        # Actualiza theta
        t0 = temp0
        t1 = temp1

        # error cuadrático medio
        e = compute_cost_function(m, t0, t1, x, y)
        st.write('J = ', e)
        J = e   # Actualiza los errores 
        iter += 1  #Actualiza las iteraciones
    
        if iter == max_iter:
            print ('Iteraciones maximas alcanzadas!')
            converged = True

    return t0,t1

def plot_cost_function(x, y, m):
    t0 = list(range(0, x.shape[0]))
    j_values = []
    for i in range(len(t0)):
        j_values.append(compute_cost_function(m, i, i, x, y)[0])
    st.write('j_values', len(j_values), len(x), len(y))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, j_values, label='curva parametrica')
    ax.legend()
    plt.show()

if __name__ == '__main__':
  

    df = pd.read_csv('./ex1data1.txt', names=['x','y'])
    x = df['x']
    y = df['y']
    
    alpha = 0.01
    ep = 0.01

    theta0, theta1 = gradient_descent(alpha, x, y, ep, max_iter=1500)
    st.write('theta0 = ' + str(theta0)+' theta1 = '+str(theta1))
  
    for i in range(x.shape[0]):
        y_predict = theta0 + theta1*x 

    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()
    st.write("Listo!")