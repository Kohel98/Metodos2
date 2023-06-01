import streamlit as st
import pandas as pd
import numpy as np

tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplos","Aplicaciones"])
with tab1:
 st.title('ProyectoMN')
 st.title(":blue[Metodo de Broyden]")

'''
El metodo cuasi Newton o método de Broyden 
Como se menciono anteriormente el metodo de Newton tiene como principal desventaja  el cálculo y evaluación de las derivadas parciales ,asi como lainversion
de la matriz jacobiana. Cuando no es práctico, o posible obtener las derivadas  parciales pueden usarse las aproximaciones  por diferencias finitas
a dichas derivadas, lo que representa  la generalizacion del método  de la secante para la solución de sistemas  de ecuaciones no lineales  y se conoce 
así como el método  de cuasi Newton o Broyden.


Este método requiere $n$ evaluaciones  funcionales por iteración  ya que remplaza la matriz jacobiana  con una matriz de aproximación 
que se actualiza en cada iteración y también disminuye el número de  calculos aritméticos a $O(n^2)$. Pertenece a una clase de tecnicas que 
reemplazan a la matriz jacobiana con una matriz de aproximación  que se actualiza con cada iteración. 
Su desventaja radica en que se pierde  la convergencia cuadrática de Newton, al ser sustituida por una convergencia denominada superlineal, 
la cual implica que:  
'''

st.latex(r"""
 \lim_{x\to\infty} (d\frac{||X^{(x+1)}-X||} {||X^{(i)}-X||}=0

""")


"""
El método de Broyden condiste en que a partir de 2 aproximaciones iniciales:


$X^{(0)}$ y $ X^{(1)}$ y la solucion de $X$ de $F(x)=0
"""

"""
se calcula de $X^{(2)}$ en afelante von el método de Broyden. La primera iteracion se calcula por método de Newton, o si es difícil determinar se calcula por $J(x^{(0)})$ se utilizan las ecuaciones de diferencias para aproximar las derivadas parciales.
Antes de iniciar con el metodo de Broyden es importante recordar que el metodo de la secante multivariable implica la siguiente fórmula para sustituir el cálculo de la derivadas.
"""

st.latex(r"""
f'(x_1)= \frac {f(x_1)- f(x_0)}{x_1-x_0}
""")

"""
Dado que en los sistemas de ecuaciones no lineales $X^{(1)}$-$X^{(0)}$ es un vector, el cociente correspondiente esta indefinido.
Sin embargo , el método procedes de manera semejante al método de Newton, por que la matriz  $J(X^{(1)})$ es reemplazada  por una matriz $A$ que tiene  la propiedad 
de que :
"""

st.latex(r"""
A^{(1)}(X^{(1)}-X^{(0)}= F(X^{(1)}-F(X^{(0)})
""")
'''

Esta matriz es la que se usa para  determinar $X^{(2)}$ como:
'''
st.latex(r"""
X^{(2)} = X^{(1)}-(A^{(1)})^{-1} F(X^{(1)})
   """)
   
"""
   Y cuyos componentes se ontienen con las dos iteraciones previas  $X^{(k)}$ y $X^{(k-1)}$ de la siguiente manera:
"""


st.latex(r"""
    A^{(k)}= A^{(k-1)} + \frac{[F(X^{(k)})-A^{(k-1)}-A^{(k-1)}(X^{(k)}-X^{(k-1)}](X^{(k)}-X^{(k-1)})^{t}}{||(X^{(k)}-X^{(k-1)})||_2^{2}}
""") 


"""
O bien 
"""

st.latex(r"""
    A^{(k)}= A^{(k-1)} + [\frac{[\Delta F^{(k)}-A^{(k-1)}-\Delta X^{(k)}](\Delta X^{(k)})^{t}}{||\Delta X^{(k)}||_2^{2}}
    """)
       
'''
 Donde
 
'''
 
st.latex(r"""
 \Delta F^{(k)}= F(X^{(k)})-F(X^{(k-1)})  y \Delta X^{(k)}=X^{(k)})-X^{(k-1)}
    """)
    
"""
  Para la primera  aplicación se requieren dos vectores iniciales $X^{(0)}$ y $X^{(1)};
  este último, como se menciono anes, puede obtenerse  con el método de Newon Raphson cuya matriz jacobiana puede emplearse para la primera 
  iteración , lo cual queda:
 """
  
st.latex(r"""  A^{(1)}= J^{(0)} + [\frac{[\Delta F^{(1)}-A^{(k-1)}-J^{(0)}\Delta X^{(1)}](\Delta X^{(1)})^{t}}{||\Delta X^{(1)}||_2^{2}}
    """)

"""
 Sin embargo, la inversión de $A^{(k)}$ en cada iteración significa un esfuerzo computacional considerable  que puede reducirse  empleando la forma de 
 inversión de Sherman Morrison.
 
 Teorema de Sherman Morrison: si A es una matriz de nxn no singular y Xy Y son vectores, entonces $A+XY^{t}$ no es singular, 
 siempre que
"""
st.latex(r""" 
 Y^{t}A^{-1}X \neq -1. 
""")
"""
Además en este caso
"""

st.latex(r"""
      (A +XY^{(t)})^{-1} = A^{(-1)}- \frac{A^{(-1)}XY^{(t)}A^{(-1)}}{1 + Y^{(t)} A^{-1}X}
    """)

"""
 Es decir, esta formula también permite calcular $(A^{(k)})^{-1}$ a partir de $(A^{(k-1)})^{-1}$, eliminando la necesidad  de invertir  una matriz en cada iteración(Burdeny Faires, 2011).
 Para ello primero se obtiene la inversa de la ecuación:
"""
  
st.latex(r"""
    (A^{(k-1)})^{-1} = (A^{(k-1)}+ [\frac{[\Delta F^{(k)}-A^{(k-1)}-\Delta X^{(k)}](\Delta X^{(k)})^{t}}{||\Delta X^{(k)}||_2^{2}}
    """)

"""
 Haciendo
"""
st.latex(r"""
    A= (A^{(k-1)}), X = [\frac{[\Delta F^{(k)}-A^{(k-1)}-\Delta X^{(k)}]}{||\Delta X^{(k)}||_2^{2}} , Y = \Delta X^{k}
    """)

"""
Se puede reescribir como: $ (A^{(k)})^{-1} = (A+XY^{(t)})^{-1} $  lo que se ajusta  el teorema, sustituyendo en la ecuación y desarrollando
"""

st.latex(r"""
    (A^{(k})^{-1} = (A^{(k-1})^{-1} + \frac{[\Delta X^{(k)}-(A^{(k-1)})^{-1} \Delta F^{(k)}] (\Delta X^{(k)})^{t}(A^{(k-1)})^{-1}}{(\Delta X^{(k)})^{t}(A^{(k-1)})^{-1} \Delta F^{(k)}}

    """)


"""
 Esta formula  permite calcular la invesa de una matriz  con sumas y multiplicaciones de matrices, con lo que se reduce el esfuerzo computacuonal al orden $n^{2}$.
 Una vez determinada  $ X {(2)} $, el método se repite  para determinar $ X{(2)} $, usando $ (A^{(2)})^{-1} $ que se obtiene a partir de (1.1) con $A^{(1)}$ y con $X^{(2)}$, y $X^{(1)}$
 en lugar de $ X^{(1)} $ y $ X^{(0)} $. En general, una vez determinado $ X^{(i)} $ se calcula $ X^{(i+1)} $ por medio de: 
 """

st.latex(r"""
    X^{(k+1)}= X^{(k)}-(A^{(k)})^{-1} F(X^{(k)}) 
  """)


"""
 Si el método  se aplica  como se describe  en las ecuaciones anteriores el numero d evaluaciones funcionales disminuye  de $n^{2}+n/a/ n$ las
 necesarias para evaluar $F(X^{(i)}$, pero todavía se requieren $O(n{3})$ cálculos para resolver el sistema lineal asociado de $nxn$
 """
 
with tab2: 
      st.title(":blue[Ejercicio:]")
 
      """
      Ejemplo: 
      Resolver  el siguiente sistema de ecuaciones no lineales empleando el método de Broyden (inversa por el teorema de Sherman Morrison).
      """                                                      
      """ Raíces (-0.8471,0.1529,1.8471),
      
         (1.1805,2.1805,-0.1805) """

st.latex(r"""
       f_1(x,y,z) = x^{2}-x + y^{2} + z^{2}-5=0  \\               
       f_2(x,y,z) = x^{2}+ y^{2} -y + z^{2}-4=0  \\                     
       f_3(x,y,z) = x^{2}+ y^{2}+ z^{2}+z -6 =0
       """) 
 
"""
Tomando como vector inicial $ X^{(0)} $ = $ \begin{equation}\begin{bmatrix} -1 \\ 0 \\ 2\ end {bmatrix}\end{equation}$, se calcula $ X^{(1)}$ por el método de Newton
"""
st.latex(r"""
\begin{equation}
X^{(1)}=
\begin{bmatrix}
-1\\
0\\
2 
\end{bmatrix}
-
\begin{bmatrix}
-3 & -0 & 4\\
-2 & -1 & 4\\
-2 & 0 & 5
\end{bmatrix}^{-1}
\begin{bmatrix}
1\\
1\\
1
\end{bmatrix}
=
 \begin{bmatrix}
-0.88571\\
0.1428\\
1.8571
\end{bmatrix}
\end{equation}
""") 
 
"""
  Con estos dos valores iniciales  se aplica la forma iterativa de Broyden 
"""
st.latex(r"""
\begin{equation}
X^{(2)}
\begin{bmatrix}
-0.8571\\
0.1428\\
1.8571 
\end{bmatrix}
-
\begin{bmatrix}
-0.7139 & -0.02174 & 0.058696\\
0.28261 & -1.02174 & 0.58696\\
-0.28261 & 0.02174 & 0.41304
\end{bmatrix}
\begin{bmatrix}
0.06122\\
0.06122\\
0.06122 
\end{bmatrix}
=
\begin{bmatrix}
-0.84783\\
0.15217\\
1.84783 
\end{bmatrix}

\end{equation}
""")

st.latex(r"""
\\
\begin{equation}
X^{(3)}
\begin{bmatrix}
-0.84783\\
0.15217\\
1.84783 
\end{bmatrix}
-
\begin{bmatrix}
-0.72118 & -0.04825 & 0.60589\\
0.27882 & -1.04825 & 0.60589\\
-0.27882 & 0.04825 & 0.39411
\end{bmatrix}
\begin{bmatrix}
0.00425\\
0.00425\\
0.00425 
\end{bmatrix}
=
\begin{bmatrix}
-0.84713\\
0.15287\\
1.84713
\end{bmatrix}
\end{equation}\\ 
""")

st.latex(r"""
\begin{equation}
X^{(4)}
\begin{bmatrix}
-0.84713\\
0.15287\\
1.84713 
\end{bmatrix}
-
\begin{bmatrix}
-0.72118 & -0.05013 & 0.60724\\
0.27882 & -1.05013 & 0.60724\\
-0.27882 & 0.05013 & 0.39276
\end{bmatrix}
\begin{bmatrix}
0.00002\\
0.00002\\
0.00002
\end{bmatrix}
=
\begin{bmatrix}
-0.84713\\
0.15287\\
1.84713
\end{bmatrix}
\end{equation}\\
""")
  
"""
Entonces,una raiz  es: 
"""
st.latex(r"""
\\
X= \begin{bmatrix}
-0.84713\\
0.15287\\
1.84713
\end{bmatrix}
""")