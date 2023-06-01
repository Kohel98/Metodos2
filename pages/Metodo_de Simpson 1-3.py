import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import *


tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplo","Aplicacion"])
with tab1:
    st.title(":blue[Regla de Simpson 1/3]")
    st.header("Definicion")
    st.markdown("Las siguientes formulas compuestas de Newton-Cotes, basadas en polinomios de interpolacion de $2^o$ y $3^{er}$ grado, son conocidas como las reglas de simpson. La primera, basada en uno cuadratico, es conocida como la regla de Simpson 1/3, y la que esta basada en un polinoio cubico se conoce como la regla de Simpson 3/8, estos nombres se deben a los coeficientes de las formulas.")
    st.markdown("La formula de $2^o$ grado de Newton-Cotes integra un polinomio cuadratico sobre dos intervalos del mismo ancho, a continuacion se construira la regla compuesta de la ecuacion")
    st.markdown(r"$\int_{x_0}^{x_1}f(x)dx = \frac{h}{3}[f_0+4f_1+2f_2]$")
    st.markdown("La formula compuesta que se aplica a una subdivision del intervalo de integracion en $n$ subintervalos (con $n$ par) es:")
    st.markdown(r"$\int_a^bf(x)dx = \frac{h}{3}[f_0+4f_1+2f_2+4f_3+2f_4+...+2f_{n-2}+4f_{n-1}+f_n]$")
    st.markdown("Esta formula se reconoce como la regla de Simpson 1/3")
    st.markdown("Cuyo termino del error global esta dado por")
    st.markdown(r"$E_{global} = -\frac{h^5}{90}\frac{(b-a)}{2h}f^{iv}(\xi) = -\frac{(b-a)}{180}h^4f^{iv}(\xi)$")
    st.markdown("Como puede verse el orden del error global cambia a $O(h^4)$. El denominador en el termino del error cambia a 180 porque se esta integrando sobre un numero de subintervalos par (significa que la regla global se aplica $h/2$ veces). El hecho que el error es $O(h^4)$, es de especial importancia")

with tab2:
    st.title(":blue[Regla de Simpson 1/3]")
    st.header("Ejemplo")
    st.markdown("Aplicar la regla de Simpson 1/3 a los datos de la siguiente tabla:")
    st.markdown(r"""
    | $i$ |   $x$  |  $f(x)$ |
    |-----|--------|---------|
    |  0  |   0.7  | 0.64835 |
    |  1  |   0.9  | 0.91360 |
    |  2  |   1.1  | 1.16092 |
    |  3  |   1.3  | 1.36178 |
    |  4  |   1.5  | 1.48500 |
    |  5  |   1.7  | 1.55007 |
    |  6  |   1.9  | 1.52882 |
    |  7  |   2.1  | 1.44513 |
    """)
    st.markdown("Dado que el numero de subintervalos es siete y no se ajusta a la regla de Simpson 1/3, existen dos posibilidades: el primero o el ultimo subintervalo se integran usando la regla del trapecio y el resto con la regla de Simpson 1/3")
    st.markdown("Primera opcion, iniciando con la regla de Simpson 1/3:")
    st.markdown(r"$\int_{0.7}^{1.9} f(x)dx \cong \frac{0.2}{3}(0.64835 +4(0.91360) + 2(1.16092) + 4(1.36178) + 2(1.48500) + 4(1.55007) + 1.52882 = 1.5193873$")
    st.markdown("Regla del trapecio:")
    st.markdown(r"$\int_{1.9}^{2.1} f(x)dx = \frac{0.2}{2}(1.52882 + 1.44513) = 0.297395$")
    st.markdown("Total:")
    st.markdown(r"$\int_{0.7}^{2.1} f(x)dx \cong 1.5193873 + 0.29739 = 1.8167823")
    st.markdown("Segunda opcion, primer intervalo por la regla del trapecio:")
    st.markdown(r"$\int_{0.7}^{0.9} f(x)dx \cong \frac{0.2}{2}(0.64835 + 0.91360) = 0.156195$")
    st.markdown("Regla de Simpson 1/3")
    st.markdown(r"$\int_{0.9}^{2.1} f(x)dx \cong \frac{0.2}{3}(0.91360 +4(1.16092) + 2(1.36178) + 4(1.48500) + 2(1.55007) + 4(1.52882) + 1.44513) = 1.661426$")
    st.markdown("Total:")
    st.markdown(r"$\int_{0.7}^{2.1} f(x)dx \cong 0.156195 + 1.661426 = 1.817621$")
    st.markdown("Normalmente no se sabe en que extremo aplicar la regla del Trapecio, por lo que graficar los puntos e identificar la forma general de la curva puede ser util.")
with tab3:
    st.title(":blue[Regla de Simpson 1/3]")
    st.header("Aplicacion")

    def regla_simpson(a, b, n, f):
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = [f.subs('x', xi) for xi in x]
        integral = 0.0
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'b-', linewidth=2)
        
        for i in range(0, n, 2):
            xi = [x[i], x[i+1], x[i+2]]
            yi = [y[i], y[i+1], y[i+2]]
            plt.fill(xi, yi, 'g', edgecolor='black', alpha=0.3)
            
            if i == n // 2:
                plt.text((x[i] + x[i+2]) / 2, max(y), f"n={n}", ha='center', va='bottom', color='r')
            
            integral += h/3 * (y[i] + 4*y[i+1] + y[i+2])
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Regla de Simpson 1/3')
        plt.grid(True)
        
        st.pyplot(plt)
        return integral

    def calcular_error(integral_exacta, integral_aproximada):
        return abs(integral_exacta - integral_aproximada)

    def main():
        st.title("Regla de Simpson 1/3 con Graficación y Error")

        st.header("Ingresar Datos")
        a = st.number_input("Valor de a:", step=0.1, format="%.2f")
        b = st.number_input("Valor de b:", step=0.1, format="%.2f")
        n = st.number_input("Número de intervalos (n):", min_value=2, step=2, value=2)
        function_str = st.text_input("Ingrese la función f(x):", value="x**2")
        integral_exacta = st.number_input("Valor de la integral exacta:", step=0.01, format="%.4f")
        
        x = symbols('x')
        try:
            f = eval(function_str)
        except:
            st.warning("La función ingresada no es válida.")
            return
        
        if n % 2 != 0:
            st.warning("El número de intervalos debe ser par para utilizar la regla de Simpson 1/3.")
            return
        
        if st.button("Calcular"):
            integral_aproximada = regla_simpson(a, b, int(n), f)
            error = calcular_error(integral_exacta, integral_aproximada)
            
            st.subheader("Resultado:")
            st.write(f"El valor aproximado de la integral es: {integral_aproximada}")
            st.write(f"El error de aproximación es: {error}")

    if __name__ == "__main__":
        main()