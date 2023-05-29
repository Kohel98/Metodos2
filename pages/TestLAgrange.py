import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly.express as ex
import plotly.graph_objects as gro
from plotly.subplots import make_subplots
tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplos","Aplicaciones"])

with tab1:
     st.title('ProyectoMN')
     st.title(":blue[Interpolación de Lagrange]")
  
     """
 Interpolación polinomial
 
 
Se tiene un conjunto de $n+1$ puntos $(x_i, y_i)$ para $i=0,1,...,n$ representar a $y$ como una función de valuación única de $x$, es posible encontrar un polinomio único de grado $n$ qué pasa por dos puntos.
Por ejemplo y es posible encontrar una recta única que pasa por dos puntos,y es posible encontrar un polinomio cuadrático único que pasa por tres puntos.

La ecuación polinomial para $y$  se puede modelar mediante:
"""
     st.latex(r"""
         y= a_0 +a_1x+a_2x^{2}+...+a_nx^{n}
         """)
     """
y los n+1 puntos se pueden usar para escribir n+1 ecuaciones para los coeficientes a_i.Estas ecuación son:
"""
     st.latex("""
y_j= a_0+a1_x_j+a3_x_j^{2}+...+a_nxj^{n} /con/ j=0,1,2,...,n
""")

     """
y constituyen un sistema de ecuaciones algebraicas lineales. La solución de este sistema se puede terminar aplicando métodos numéricos para ese fin. 
Sin embargo, la matriz de coeficientes, denominada matriz de vander homonde, es sensible al mal planeamiento. 
Además, sobre el sistema de esa manera ineficaz de obtener una representación para $y$

Por lo anterior, se opta por otros métodos por lineales que representan formas más eficaces de predecir y para un valor dado de $x$.
La apariencia de estos modelos puede ser muy distinta a la del modelo antes mencionado; sin embargo, producen la misma curva única que pasa por los $n$ puntos.

Fórmula de lagrange
*Es una técnica que permite encontrar el máximo o mínimo de una función de varias 
dimensiones cuando hay alguna restricción en los valores de entrada que puede usar.


Los polinomios de lagrange se pueden determinar especificando algunos de los puntos en el plano por los cuales debe pasar.
Considérese el problema de determinar el polinomio de grado uno que pasa por los puntos distintos. 

(x_0,y_0) y (x_1,x_2). Este problema es el mismo que el de aproximar una función $f$,para la cual 
$f(x_0)$ =y_0 y fx_1=y_1 por medio de un polinomio de primer grado, interpolando entre,o conincidiendo con,
los valores de $f(x)$ en los puntos dados.
"""
     """
considerese el polinomio 

"""

     st.latex(r""" 
L_0(x)=\frac{ (x-x_1)}{(x_0-x_1)}
L_1(x)\frac{ (x-x_0)}{(x_1-x_0)}
P(x)=  L_0(x)y_0+L_1(x)y_1
""")
     """
el polinomio lineal que pasa por  $ (x_0,f(_0))$ y$ (x_1,f(_1)) $ se construye usando los cocientes
"""

     st.latex(r"""L_0(x)=\frac{ (x-x_1)}{(x_0-x_1)}
L_1(x)\frac{ (x-x_0)}{(x_1-x_0)}\\

P(x)=  L_0(x)y_0+L_1(x)y_1
""")
     """
cuando $x=x_0$
"""
     st.latex(r""" 
P(x)=\frac{ (x_0-x_1)}{(x_0-x_1)}y_0 +\frac{ (x_0-x_0)}{(x_1-x_0)}y_1 =y_0=f(x_0)
""")
     """
$L_0(x_0)=1$, mientras que $L_1(x_0)=0$
"""
     """
cuando $x=x1$
"""
     st.latex(r""" 
P(x)=\frac{ (x_1-x_1)}{(x_0-x_1)}y_0 +\frac{ (x_1-x_0)}{(x_1-x_0)}y_1 =y_1=f(x_1)\\
L_0(x_1)=0 y L_1(x_1)=1
""")

     """
Para generalizar el concepto de interpolación lineal, considérese la construcción de un polinomio a lo más grado $n$ que pase por los $n+1$ puntos $(x_0,f(x_0))$, $(x_1,f(x_1))$,$ (x_2,f(x_2))$, …, $(x_n,f(x_n))$, 
Para lo que se quiere, para cada $k=0,1…, n$, un cociente $L_k(x)$ con la propiedad de que $L_k(x_i) =0$ Cuando $i$ distinto de $k$, por lo que el numerador debe contener el término:
"""
     st.latex(r"""
(x-x_0)(x-x_1)...(x-x_k-1)(x-x_k+1)...(x-x_n)
""")


     """"
para satisfacer que $ L_k(X_k) =1 $, el denominador debe ser igual al numerador anterior cuando $ x=x_k $, por lo que cocientes tienen la forma:


para cada $k=0,1…, n$; el cual se denomina el cociente de lagrange. A partir del cual se define el polinomio de interpolación de lagrange en el siguiente teorema.
Teorema
sí $x _o, x_1,…,x_n$ son $(n+1)4 son números diferentes y $f(x)$ es una función cuyos valores están dados en estos puntos, entonces existe un único polinomioP(x) de grado a lo más n con la propiedad de que f(x_k)=P(x_k) hd para cada k=0,1…, n.
Este polinomio está dado por

Dónde $L_k(x)$ está dado por la ecuación del teorema anterior
la técnica usada para construir $P(x)$ esto de interpolación hola que se empleaba para construir las tablas trigonométricas con logarítmicas. 

Ejemplo
las densidades de sodio para 3 temperaturas están dadas por

hola para determinar la densidad para t=251°C
dado que se tienen 3 puntos, el polinomio que se puede construir es de a lo más  de grado dos, para obtener el valor de los cocientes, se sustituye el valor de t=251

entonces, la densidad del sodio a una temperatura t=251°C es de d= 890.5566117 Kg/m^{3}

aquí  cabe aclarar que, aunque con el método de lagrange se puede obtener una expresión ese que potencias que aproxime a la función que describe la table datos, esto no es una práctica común, porque generalmente se aplica mediante un programa computadora y porque existen métodos más eficientes para este fin.
"""
with tab2:
    """_summary_

    Returns:
        _type_: _description_
    """
with tab3:

    def get_sympy_subplots(plot:Plot):
        """
        It takes a plot object and returns a matplotlib figure object

        :param plot: The plot object to be rendered
        :type plot: Plot
        :return: A matplotlib figure object.
        """
        backend = MatplotlibBackend(plot)

        backend.process_series()
        backend.fig.tight_layout()
        return backend.plt

    def li(v, i):
        """
        The function takes a list of numbers and an index, and returns the Lagrange interpolating polynomial for the list of
        numbers with the index'th number removed

        :param v: the list of x values
        :param i: the index of the x value you want to interpolate
        :return: the Lagrange interpolating polynomial for the given data points.
        """
        x = sy.symbols('x')

        s = 1
        st = ''
        for k in range(0,len(v)):
            if k != i:
                st = st + '((' + str(x) + '-'+ str(v[k])+')/('+str(v[i])+'-'+str(v[k])+'))'
                s = s*((x-v[k])/(v[i]-v[k]))

        return s

    def Lagrange(v,fx):
        """
        It takes in a list of x values and a list of y values, and returns the Lagrange polynomial that interpolates those
        points

        :param v: list of x values
        :param fx: The function you want to interpolate
        :return: the Lagrange polynomial.
        """
        #print(v)
        #print(fx)
        lis = []
        for i in range(0,len(v)):
            lis.append(li(v,i))

        sums = 0

        for k in range(0,len(v)):
            sums = sums+(fx[k]*lis[k])

        #print(sums)

        sy.simplify(sums)

        sy.pprint(sums)

        p1 = sy.plot(sums,show=False)
        p2 = get_sympy_subplots(p1)
        p2.plot(v,fx,"o")
        #p2.show()
        return sy.expand(sums), p2,lis

    st.title(':blue[Interpolación de Lagrange]')

    st.subheader(':blue[Descripción del método]')

    st.subheader(':blue[Ejemplo]')


    st.subheader('Método')

    filess = st.sidebar.file_uploader('Selecciona un archivo de prueba: ')
    if filess != None:
        fi = pd.read_csv(filess)
        st.write('Los datos a interpolar son: ')
        st.write(fi)
        x = list(fi['x'])
        fx = list(fi['y'])
    else:
        xxs = st.text_input('Ingrese los valores de $x_k$: ',value='{1,2,3,4}')

        xsstr = ''


        for i in xxs:

            if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
                xsstr = xsstr + i

        fxxs = st.text_input('Ingrese los valores de $f(x_k)$: ',value='{-1,3,4,5}')

        x = list(map(float,xsstr.split(',')))
        intstrr = ''




        for t in fxxs:

            if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
                intstrr = intstrr + t

        fx = list(map(float,intstrr.split(',')))


    #st.write(x)
    #st.write(fx)
    #data = [x,fx]
    #st.write(data)


    method = Lagrange(x,fx)

    st.write('_Los polinomios fundamentales de Lagrange estan dados por:_')
    lli = r'''l_i(x) = \begin{cases}'''
    for t in range(0,len(method[2])):
        lli = lli +'l_'+str(t)+r'='+sy.latex(sy.expand(method[2][t]))+r'\\'
    lli = lli + r'\end{cases}'
    st.latex(lli)
    st.write('_El polinomio de Interpolacion está dado por:_')
    st.latex(r'p_n(x) = \sum_{i=0}^{n} l_i(x)f(x_i)')
    st.latex('p_n(x) =' + sy.latex(method[0]))

    func = sy.lambdify(sy.symbols('x'),method[0])
    funcdata = pd.DataFrame(dict(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000))))

    plo = gro.Figure()

    plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000)),name='Interpolación'))
    plo.add_trace(gro.Scatter(x=x,y=fx, marker_color='rgba(152, 0, 0, .8)',name='Datos'))
    #plo.add_hline(y=0)
    #plo.add_vline(x=0)
    plo.update_layout(title='Grafica de la Interpolación')
    st.plotly_chart(plo)

