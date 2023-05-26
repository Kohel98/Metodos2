
import streamlit as st
import pandas as pd
import numpy as np
tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplos","Aplicaciones"])
with tab1:
         st.title(":blue[Metodo de Newton(-Raphson)]")
         st.header("Teorema")
         """
         El metodo de Newton es una alternativa del metodo del punto 
         fijo ya que la gran desventaja de este se consiste en obtener ecuaciones $g_i$ que converja 
         en el intervalo de solucion y dicha búsqueda puede representar un esfuerzo de cálculo considerable, en lo unico en que se parecen en estos dos metodos es 
         la generalización multivariable del método de Newtoon para una sola variable.
         Este metodo solo sirve para los sistemas de ecuaciones no lineales, este requiere algunos conceptos previos, los cuales se presentan a continuación.
         La derivada, cuando se trabaja con funciones con varias variables, se emplean las derivadas parciales. La generalización de derivada para sistemas de funciones 
         de varias variables es la matriz jacobiana. \\
         :red[**Definición**]. Matriz jacobiana: Sean $f_i (x_1, x_2, ..., x_n)$ con $ 1 \leq i \leq n $, funciones con n variables $(x_i)$ independientes. La matriz jacobiana $J(x_1, x_2,..., x_n)$
         está dada por las derivadas parciales de cada una de las funciones con respecto a cada una de las variables: 
         """
         "Matriz Jacobiana (J):"
         st.latex(r"""
                  \begin{bmatrix}
                  \dfrac{\partial f_1}{\partial x_1} (x_0, x_1, ..., x_n) & \dfrac{\partial f_1}{\partial x_2} (x_0, x_1, ..., x_n)&... ... & \dfrac{\partial f_1}{\partial x_n} (x_0, x_1, ..., x_n) \\
                  & & &\\
                  \dfrac{\partial f_2}{\partial x_1} (x_0, x_1, ..., x_n) & \dfrac{\partial f_2}{\partial x_2} (x_0, x_1, ..., x_n)&... ...  &\dfrac{\partial f_2}{\partial x_n} (x_0, x_1, ..., x_n) \\
                  ... & & & \\
                  \dfrac{\partial f_n}{\partial x_1} (x_0, x_1, ..., x_n) & \dfrac{\partial f_n}{\partial x_2} (x_0, x_1, ..., x_n)&... ...& \dfrac{\partial f_n}{\partial x_n} (x_0, x_1, ..., x_n)
                  \end{bmatrix}
                  """)
         st.latex(r"""
                  \begin{bmatrix} 
                  f_{1 x_1} & f_{1 x_2}&... ... & f_{1 x_n} \\
                  & & &\\
                  f_{2 x_1} & f_{2 x_1}&... ...  &f_{2 x_n} \\
                  ... & & & \\
                  f_{n x_1} & f_{n x_2}&... ...& f_{n x_n}
                  \end{bmatrix}
                  """)
         """
         El método de Newton para una variable se basa en la expansión de la serie de Taylor de primer orden:\\
         $f(x_{i+1}) = f(x_i) + (x_{i+1} - x_i) f'(x_i)$\\
         Donde $x_i$ es el valor inicial de la raíz y $x_{i+1}$ es el punto en el cual la derivada (pendiente) interseca del eje. En esta intersección 
         $f(x_{i+1})$ por definición es igual a cero y la forma iterativa del método puede escribirse como:
         """
         """   $x_{i+1} = x_i - \dfrac{f(x_i)}{f'(x_i)}$ """
         """
         La forma 'simple' de la ecuación del método de Newton.
         La forma para varias ecuaciones se deriva en forma idéntica, pero a partir de la serie de Taylor para varias variables.
         Escribiendo esta ecuación en forma matricial: 
                   + Para el sistema $f_n(x_1, x_2,... x_n)$ se tiene: $F(X)=0$
                   + Definiendo los vectores columna como \\
                   $ F = (f_{1}, f_{2}, ..., f_{n})^{t},$ $ X = (x_{1}, x_{2}, x_{n})^{t}$
                   + La extensión del método de Newton para sistemas no lineales es : \\
                   $X^{(k+1)} = X^{(k)} - [F'(X^{(k)}]^{-1} F(X^{(k)})$ \\
                   Donde $F'(x^ {(k)})$ es la matriz jacobiana: \\
                   $X^{(k+1)} = X^{(K)} - J(X^{(k)})^{-1} F(X^{(k)})$ \\
                   Esta ecuación es análoga a la del método de Newton-Raphson para una ecuación, sólo que la derivada aparece en el numerador como la inversa de la matriz jacobiana.
                   La ecuacion $X^{(k+1)} = X^{(K)} - J(X^{(k)})^{-1} F(X^{(k)})$ es la representación del método de Newton-Raphson para sistemas de no lineales y generalmente se espera que 
                   dé una convergencia cuadratica, siempre y cuando se conozca un valor inicial suficiententemente preciso y exista $J(X^{k})^{-1}$ \\
                   
                   """
         with tab2: 
                  st.title(":blue[Ejercicio:]")
                  """
                   Sea el siguiente sistema de ecuaciones: \\
                           + $f_{1} (x,y) = 4-x^{2} -y^{2} = 0 $ \\
                           + $f_{2} (x,y) = 1-e^{x} -y = 0 $
                  """
                  """
                  Se obtienen las derivadas parciales para escribir la matriz jacobiana: 
                  """
                  st.latex(r"""
                  \begin{matrix} 
                  f_{1x} = -2x & f_{1y} = -2y \\
                  & \\
                  f_{2x} = -e^{x} &  f_{2y} = -1 
               
                  \end{matrix}
                  """)
                  """
                  Jacobiana J(X)
                  """
                  st.latex(r"""
                  \begin{matrix} 
                  f_{1x} = -2x & f_{1y} = -2y \\
                  & \\
                  f_{2x} = -e^{x} &  f_{2y} = -1 
               
                  \end{matrix}
                  """)
                  """
                  Se eligen valores iniciales para la aproximación $X^{(0)} = (1,-1.7)^{t}$ , como valor de $x_0$ y $y_0$ que se sustituyen en $X^{(k+1)} = X^{(K)} - J(X^{(k)})^{-1} F(X^{(k)})$
                  """
                  st.latex(r""" X^{(1)} = 
                  \begin{bmatrix} 
                  1\\
                   \\
                  -1.7
                  \end{bmatrix}
                  -
                  \begin{bmatrix} 
                  -2 & 3.4 \\
                  & \\
                 -2.7183 &  -1.0 
               
                  \end{bmatrix} ^{-1}
                  
                  \begin{bmatrix} 
                  0.1100\\
                   \\
                  -0.0183
                  \end{bmatrix}
                  =
                  \begin{bmatrix} 
                  1.00426\\
                   \\
                  -1.72985
                  \end{bmatrix}
                  """)
                  """
                  Estos valores tienen una aproximacion optima pero aun se puede mejorar, para esto requiere otra iteracion para mejorarla. 
                  """
                  st.latex(r""" X^{(2)} = 
                  \begin{bmatrix} 
                  1.00426\\
                   \\
                  -1.72985
                  \end{bmatrix}
                  -
                  \begin{bmatrix} 
                  -2.00851 & 3.4597 \\
                  & \\
                 -2.72987 &  -1.0 
               
                  \end{bmatrix} ^{-1}
                  
                  \begin{bmatrix} 
                  -0.0091\\
                   \\
                  -0.00002
                  \end{bmatrix}
                  =
                  \begin{bmatrix} 
                  1.00417\\
                   \\
                  -1.72963
                  \end{bmatrix}
                  """)
                  """
                  Las iteraciones se detienen cuando la norma espectral del vector $F(X)$ es menor a una tolerancia previamente establecida. \\
                  \\
                  Si se cambia el punto inicial a $X^{(0)} = (-1.5, 0.6)^{t}$ se obtiene la segunda raíz $X=(-1.816264, 0.837367) ^{t}$.\\
                  \\
                  Es importante calcular la matriz jacobiana y su inversa en cada interación. 
                  """
                  
