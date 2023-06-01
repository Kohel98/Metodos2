import streamlit as st
import pandas as pd
import numpy as np

tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplos","Aplicaciones"])
with tab1:
  st.title('ProyectoMN')
  st.title(":blue[**Método de Diferencias Divididas**]")

"""
  Fórmula de diferencias divididas
  Los métodos para determinar la representación explícita de un polinomio de interpolación, a partir de datos tabulados, se conocen como diferencias y pueden usarse con de deducir técnicas para aproximar las derivadas y las integrales de las funciones, 
  así como para aproximar las soluciones de ecuaciones diferenciales. 
  El tratamiento de las tablas de diferencias divididas supone que la función $f(x)$ es conocida para varios valores de x. Dichos valores no necesariamente están igualmente espaciados u obedecen algún orden (sin embargo, si están ordenados puede ser ventajoso)

  Supóngase que $P_n(x)$ ese polinomio de lagrange de grado a lo más $n$ que coincide con la función $f(x)$ en los números distintos $x _0, x_1,…,x_n$. Las diferencias divididas $f(x)$ con respecto  $a x _0, x_1,…,x_n$ 
  se pueden deducir demostrando que $P_n(x)$ tiene representación
"""
st.latex(r"""
P_n(x)= a_0 +a_1(x-x_o)+a_2(x-x_0)(x-x_1)+ ...+a_n(x-x_0)(x-x_1)...(x-x_n-1)   (2.1)
""")

"""
  Con constantes apropiadas $a _0, a_1,…,a_n$
  Si $P_n(k)$ cumple con la condición de pasar por los puntos empleados en su construcción, entonces evaluando en la ecuación 
  anterior $x=x_0$ queda: 
"""
st.latex(r"""
 P_n(x_0)= a_0 =f(x_0) 
 """)
  
""" 
   Similarmente , cuand se evalúa en x_1, los unicos términos distintos de cero en la evaluación  de $P_n(x_1)$ son la constante y el término lineal, 
   el cual se despeja $a_1$:
   
"""
st.latex(r"""
  P_n(x_1)=f(x_0)+a_1(x_1-x_0)=f(x_1)
 """)
  
st.latex(r"""
  a_1= \frac {f(x_1)-f(x_0)}{x_1-x_0}
 """)
"""
  Es aqui en donde se  introduce la notación de la diferencia dividida, la primera diferencia dividida de $f(x)$ con respecto a $x_i$ y $x_i+1$, se denota  por $f[x_i,X_i+1]$ 
  y está definida  como:
"""
st.latex(r"""
  f[x_i,x_i+1]= \frac{f[x_i+1]-f[x_i]}{x_i+1 -x_i}
 """)
"""
  La diferencia dividida de cero de la función $f(x)$, con respecto a $x_i$, se denota por $f[x_i]$, y es simplemente  la evaluación de $f(x)$ en $x_i$
"""
st.latex(r"""
  f[x_i]= f(x_i) 
 """)
  
"""
  Las diferencias divididas se definen  inductivamente , las pimeras de las cuales se muestran  en la siguiente tabla
"""
  
  
"""  
| $x_i$           | $f_i=f[x_i]$    | $f[x_i,x_i+1]$ |$ f[x_i,x_i+1,x_i+2]$ |$ f[x_i,x_i+1,x_i+2, x_i+3]$|
| --------------| ------------- |--------------|--------------------| -----------------|
|   $x_0$ | $f_0$ |  $f[x_0,x_1]$= $ \dfrac {f_1-f_0}{x_1-x_0}$ | $f[x_0,x_1,x_2]$= $\dfrac {f[x_1-x_2]-f[x_0,x_1]}{x_2-x_0}$ |$ f[X_0,x_1,X_2, X_3] $ $\ frac {f[X_0,x_1,X_2, X_3]-f[X_0,x_1,X_2}{x_3 -x_0}$
|   $x_1$ | $f_1$ | $ f[x_1,x_2]$ = $ \dfrac {f_2-f_0}{x_2-x_1}$ | $f[x_1,x_2,x_3]$= $\dfrac{f[x_2,x_3]-f[x_1,x_2]}{x_3-x_1}$ | $f[X_0,x_1,X_2, X_3, X_4]$
|   $x_2$ | $f_2$ |  $f[x_2,x_3] $= $ \dfrac {f_3-f_2}{x_3-x_2} $| $f[x_2,x_3,x_4]=$ $\dfrac {f[x_3,x_4]-f[x_2,x_3]}{x_4-x_2}$ | 
|   $x_3$ | $f_3$ |  $f[x_3,x_4]$= $\dfrac {f_4-f_3}{x_4-x_3}$ |
|   $x_4$ | $f_4$ | 
"""
"""
Cuando se han determinado las (k-1) diferencias divididas
"""
st.latex(r"""
  f[x_1,x_i+1, x_i+2,...,x_i+k-1] y  f[x_1,x_i+1, x_i+2,...,x_i+k]
 """)
"""
La $k-ésima$ diferencia dividida de $f$ relativa a x_i,X_i+1, X_i+2,...,x_i+k esta dada por:
"""
st.latex(r"""
  f[x_1,x_i+1, x_i+2,...,x_i+k]= \frac {f[x_1,x_i+1, x_i+2,...,x_i+k]- f[x_1,X_i+1, x_i+2,...,x_i+k-1]}{x_i+k-x_i}
 """)
"""
Estas diferencias divididas  corresponde a las cosnstantes requerids para cada k=0,1,2,...,n;así el polinomio (2.1) piede reescribirse coomo:
"""
st.latex(r"""
P_n(x)= f[x_0]+ f[x_0,x_1](x-x_0)+ f[x_0,x_1,x_2](x-x_0)+a_2(x-x_0)(x-x_1)+ ...
f[x_0,x_1,...,x_n](x-x_0)(x-x_1)...(x-x_n-1) 
""")
st.latex(r"""
P_n(x)= f[x_0]+ \[\sum_{k=1}^n f[x_0,x_1,...,x_n] (x-x_0)(x-x_1)...(x-X_k-1)
""")
"""
A esta última se le conoce como la fórmula de diferencia dividida interpolante de Newton. 
"""
with tab2:
    """
    **Ejercicio** Sea $f(x)=3^x$ con $x∈R$. Sea $p(x)$ el polinomio de grado a lo mas dos que concuerdas con la función en los puntos $x_0=0,x_1=1$ y 
    $x_2=2$. usando diferencias divididas para construir a p(x).
    """
    """
    ssss
     Para construir el polinomio $p(x)$ de grado a lo más dos que concuerda con la función $f(x)=3x$ en los puntos $x_0=0, x_1=1$ y $x_2=2$ usando el método de diferencias divididas, seguimos los siguientes pasos: 

      1)    Calculamos las diferencias divididas de primer orden 
     $f[x_0,x1] = \dfrac{f(x_1)-f(x_0)}{x_1 -x_0} =$ $\dfrac{3-1}{1-0} = 2$
     $f[x_1,x2] = \dfrac{f(x_2)-f(x_1)}{x_2 -x_1} =$ $\dfrac{9-3}{2-1} = 6$

    2) Calculamos las diferencias divididas de segundo orden: \
    $f[x_0, x_1, x_2]$=$\dfrac{f[x_1,x_2]-f[x_0,x_1]}{x_2-x_0}$ = $\dfrac{6-2}{9-1}$ = $\dfrac{4}{8}$ = $\dfrac{1}{2}$ \

    3)Usando la fórmula de interpolación de Newton para el polinomio $p(x)$, tenemos: \
    $p(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1)$ \

    Sustituyendo los valores obtenidos de las diferencias divididas: \

    $p(x) = 1 + 2x + (\dfrac{1}{2} x)$ $(\dfrac{1}{2} x - \dfrac{1}{2} )$  \
     = $\dfrac {x^2}{4}$ - $\dfrac {x}{4} + 1 + 2x $  
    """
    
    
    
    
    
    
    