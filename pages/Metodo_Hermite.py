import streamlit as st
import pandas as pd
import numpy as np


tab1, tab2, tab3 = st.tabs(["Definiciones","Ejemplo","Aplicacion"])
with tab1:
    st.title(":blue[Metodo de Hermite]")
    st.header("Definicion")
    st.markdown("Un polinomio se puede ajustar no solo a los valores de la función sino también a las derivadas en los puntos. Los polinomios ajustados a los valores de la función y de su derivada se llaman *polinomios de interpolación de Hermite* o *polinomios osculantes*.")
    st.markdown("El conjunto de los polinomios osculantes es una generalización de los polinomios de Taylor y los polinomios de Lagrange. Estos polinomios tienen la propiedad de que dados $n + 1$ números distintos $x_0, x_1, x_2, ..., x_n$ y los enteros no negativos $m_0, m_1, m_2, ..., m_n$, el polinomio osculante que aproxima a una función $f \in C^m[a, b]$, donde $m = max(m_0, m_1, m_2, ..., m_n)$ y $x_i \in [a, b]$ para cada $i = 0, 1, ..., n$, es el polinomio de menor grado con la propiedad de que coincide con la función f y todas sus derivadas de orden menor o igual a $m_i$ en $x_i$ para cada $i = 0, 1, ..., n$. El grado de este polinomio osculante será a lo más")
    st.markdown(r"M = $$\sum_{i=0}^n m_i + n$$")
    st.markdown("Ya que el numero de condiciones que se tiene que satisfacer es $$\sum_{i=0}^n m_i + (n + 1)$$ y un polinomio de grado $M$ tiene $M + 1$ coeficientes que pueden usarse para satisfacer estas condiciones.")
    st.subheader("Defincion 2")
    st.markdown("Sean $x_0, x_1, x_2, ..., x_n, n+1$ numeros distintos en $[a,b]$ y $m_i$ un entero no-negativo asociado a $x_i$ para $i = 0, 1, ..., n$. Supongase que $f \in C^m[a, b]$ y que $m = max_{0 \leq i \leq n} m_i$. El plinomio osculante que aproxima a $f(x)$ es el polinomio $P(x)$ de menor grado tal que")
    st.markdown(r"$\frac{d^k P(x_i)}{dx^k} = \frac{d^k f(x_i)}{dx^k}$     Para $i = 0, 1, ..., n$ y $k = 0, 1, 2, ..., m$.")
    st.markdown("Notese que se cuando $n = 0$ el polinomio osculante que aproxima a $f(x)$ es el polinomio de Taylor de grado $m_0$ para $f(x)$ en $x_0$. Cuando $m_i = 1$ para cada $i = 0, 1, ..., n$ da una clase de polinomios llamados polinomios de Hermite. Para una funcion dada $f$, estos polinomios no solo coinciden con $f$ en $x_0, x_1, x_2, ..., x_n$, sino que, como sus primeras derivadas coinciden tambien con las de $f$, tienen la misma apariencia que la funcion en $(x_i, f(x_i))$ en el sentido que las lineas tangentes al polinomio y a la funcion coinciden.")
    st.subheader("Teorema 1")
    st.markdown("Si $f \in C^1[a, b]$ y $x_0, x_1, x_2, ..., x_n \in [a,b]$ son distintos, el unico polinomio de menor grado que coincide con $f$ y $f'$ en $x_0, x_1, x_2, ..., x_n$ es un polinomio de grado a lo mas $2n + 1$ dado por")
    st.markdown(r"$H_{2n+1}(x) = \sum_{i=0}^n f(x_j)H_{n,j}(x) + \sum_{i=0}^n f'(x_j)\widehat(H_{n,j})(x)$")
    st.markdown("Donde:")
    st.markdown(r"$H_{n,j}(x) = [1-2(x-x_j)L'_{n,j}(x_j)]L_{n,j}^2 (x)$")
    st.markdown(r"$\widehat(H_{n,j})(x) = (x-x_j)L_{n,j}^2 (x)$")
    st.markdown("En este contexto, $L_{n,j}$ denota al j-esimo coeficiente polinomial de Lagrange de grado $n$, definido en la ecuacion")
    st.markdown(r"$L_{n,j}(x) = \prod_{i=1, i \neq j}^n \frac{x-x_i}{x_j - x_i}$ para cada $j = 0, 1, ..., n$")
    st.markdown("Ademas, si $f \in C^{(2n+2)}[a, b]$, entonces")
    st.markdown(r"$f(x)-H_{2n+1}(x) = \frac{(x-x_0)^2 ... (x-x_n)^2}{(2n+2)!}f^{(2n+2)}(\xi)$")
    st.markdown(r"Para alguna $\xi$ con $a<\xi<b$.")
with tab2:
    st.title(":blue[Metodo de Hermite]")
    st.header("Ejemplo")
    st.markdown("Usar el polinomio de menor grado que coincide con los datos mostrados en la tabla para la funcion de Bessel de pimera clase de orden cero para encontrar una aproximacion de $f(1.5)$")
    st. markdown("""
    | $j$ | $x_j$ |  $f(x_j)$   |  $f'(x_j)$   |
    |-----|-------|-------------|--------------|
    |  0  |  1.3  |  0.6200860  |  -0.5220232  |
    |  1  |  1.6  |  0.4554022  |  -0.5698959  |
    |  2  |  1.9  |  0.2818186  |  -0.5811571  |
    """)
    st.markdown("Primero se calculan los polinomios de Lagrange y sus derivadas, para poder calcular los componentes del polinomio. Los resultados se muestran en la siguiente tabla:")
    st.markdown("""
    | $i$ | $x_j$ | $f(x_j)$    | $f'(x_j)$    | $L_j(x)$ | $L_j(x_j)$ | $H_{2,j}$ | $H_{2,j}$   |
    |-----|-------|-------------|--------------|----------|------------|-----------|-------------|
    |  0  |  1.3  |  0.6200860  |  -0.5220232  |   2/9    |    -5      |    4/27   |     4/405   |
    |  1  |  1.6  |  0.4554022  |  -0.5698959  |   8/9    |     0      |   64/81   |   -32/405   |
    |  2  |  1.9  |  0.2818186  |  -0.5811571  |  -1/9    |     5      |    5/81   |    -2/405   |
    """)
    st.markdown("Entonces el polinomio queda:")
    st.markdown(r"$H_5(1.5) = 0.620086(\frac{4}{27})+(\frac{64}{0.455402281})+0.2818186(\frac{5}{81})-0.5220232(\frac{4}{405})-0.5698959(\frac{-32}{405})-0.5811571(\frac{-2}{405})=0.511827701727$")
    st.markdown("Este resultado es exacto en todas las cifras dadas.")
    st.markdown("La necesidad de determinar y evaluar los cocientes de Lagrange y sus derivadas, asi como el calculo de los componentes del polinomio, hace el procedimiento complicado y suceptible de errores, aun para valores pequeños de $n$. Un metodo alternativo para generar aproximaciones de Hermite esta basado en la formula de diferencias divididas de Newton.")
    st.markdown(r"$P_n(x) = f[x_0] + \sum_{k=1}^n f[x_0, x_1, ..., x_k](x-x_0)(x-x_1)...(x-x_{k-1})$")
    st.markdown("y la conexion entre la $n$-esima diferencia dividida y la $n$-esima derivada de $f(x)$")
    st.markdown("Supongase que se dan $n + 1$ numeros distintos $x_0, x_1, ..., x_n$ con sus valores de $f(x)$ y $f'(x)$. Lo primero es definir una nueva sucesion $z_0, z_1, ..., z_{2n+1}$ por")
    st.markdown("$z_{2i} = z_{2i+1} = x_i$ para cada $i = 0, 1, ..., n$.")
    st.markdown("Con esto se contruy la tabla de diferencias divididas de la forma en que se ha venido realizando, pero ultilizando $z_0, z_1, ..., z_{2n+1}$.")
    st.markdown("Como cada $z_{2i} = z_{2i+1} = x_i$ para cada $i$, $f[z_{2i}, z_{2i+1}]$ no puede ser definida por la relacion basica")
    st.markdown(r"$f[x_i, x_{i+1}]= \frac{f[x_{i+1}]-f[x_i]}{x_{i+1}-x_i}$")
    st.markdown("Sin embargo, si se supone que la sustitucion razonable en esta situacion es:")
    st.markdown(r"$f[z_{2i},z_{2i+1}]=f'(x_i)$")
    st.markdown("Se puede usar los valores $f'(x_0), f'(x_1), ..., f'(x_n)$ en lugar de las primeras diferencias divididas indefinidas.")
    st.markdown(r"""
    |        $z$       |          $f(z)$        |            $f(z_i, z_{i+1})$                 |
    |------------------|------------------------|----------------------------------------------|
    |    $z_0 = x_0$   |    $f[z_0] = f(x_0)$   |           $f[z_0,z_1] = f'(x_0)$             |
    |    $z_1 = x_0$   |    $f[z_1] = f(x_0)$   | $f[z_1,z_2] = \frac{f[z_2]-f[z_1]}{z_2-z_1}$ |
    |    $z_2 = x_1$   |    $f[z_2] = f(x_1)$   |           $f[z_0,z_1] = f'(x_1)$             |
    |    $z_3 = x_1$   |    $f[z_2] = f(x_1)$   | $f[z_0,z_1] = \frac{f[z_4]-f[z_3]}{z_4-z_3}$ |
    |        ...       |           ...          |                     ...                      |
    |   $z_{2n} = x_n$ |  $f[z_{2n}] = f(x_n)$  |        $f[z_{2n},z_{2n+1}] = f'(x_n)$        |
    | $z_{2n+1} = x_n$ | $f[z_{2n+1}] = f(x_n)$ |                                              |
    """)
    st.markdown("Las diferencias divididas restantes se producen de la manera usual y por ultimo se emplean las diferencias divididas apropiadas para construir el polinomio de interpolacion de Newton.")
    st.markdown(r"""
    | $z_i$ |   $f(z_i)$  |  $f(z_i,z_{i+1})$   | $f(z_i,z_{i+1},z_{i+2})$ | $f(z_i,z_{i+1},z_{i+2},z_{i+3})$ | $f_i^[4]$ | $f_i^[5]$ |
    |-------|-------------|---------------------|--------------------------|----------------------------------|-----------|-----------|
    |  1.3  |  0.6200860  |     -0.5220232      |        -0.0897427        |            0.0663657             | 0.0026663 | -0.0027738|
    |  1.3  |  0.6200860  |     -0.5489460      |        -0.0698330        |            0.0679655             | 0.0010020 |           |
    |  1.6  |  0.4554022  |     -0.5698959      |        -0.0290537        |            0.0685667             |           |           |
    |  1.6  |  0.4554022  |     -0.5786120      |        -0.0084837        |                                  |           |           |
    |  1.9  |  0.2818186  |     -0.5811571      |                          |                                  |           |           |
    |  1.9  |  0.2818186  |                     |                          |                                  |           |           |
    """)
    st.markdown(r"$H_5(1.5) = 0.620086 + (1.5-1.3)(-0.5220232) + (1.5-1.3)^2(-0.0897427) + (1.5-1.3)^2(1.5-1.6)( 0.0663657) + (1.5-1.3)^2(1.5-1.6)^2(0.0026663) + (1.5-1.3)^2(1.5-1.6)^2(1.5-1.9)(-0.0027738) = 0.5118277$")
    st.markdown("Como puede observarse el resultado es el mismo, lo que puede proporcionar una alternativa viable para su automatizacion con un programa.")
with tab3:
    st.title(":blue[Metodo de Hermite]")
    st.header("Aplicacion")

    def hermite_interpolation(x, f_x, f_prime_x, x_interp):
        n = len(x)
        m = 2 * n
        
        Q = np.zeros((m, m))
        b = np.zeros(m)
        
        for i in range(n):
            Q[2 * i][0] = x[i]
            Q[2 * i + 1][0] = x[i]
            Q[2 * i + 1][1] = f_prime_x[i]
            
            b[2 * i] = f_x[i]
            b[2 * i + 1] = f_x[i]
        
        for i in range(2, m):
            for j in range(1, i):
                Q[i][j] = (Q[i][j-1] - Q[i-1][j-1]) / (Q[i][0] - Q[i-j][0])
        
        interpolated_values = []
        
        for interp_point in x_interp:
            p = Q[0][0]
            product = 1
            
            for i in range(1, m):
                product *= (interp_point - Q[i-1][0])
                p += Q[i][i] * product
            
            interpolated_values.append(p)
        
        return interpolated_values


    def main():
        st.title("Método de Hermite")
        
        st.header("Ingresar Datos")
        n = st.number_input("Número de puntos:", min_value=2, step=1, value=3)
        
        x_values = []
        f_x_values = []
        f_prime_x_values = []
        
        for i in range(n):
            st.subheader(f"Punto {i+1}")
            x = st.number_input("Valor de x:", key=f"valor_x_{i}")
            f_x = st.number_input("Valor de f(x):", key=f"valor_y_{i}")
            f_prime_x = st.number_input("Valor de f'(x):", key=f"valor_fx_{i}")
            
            x_values.append(x)
            f_x_values.append(f_x)
            f_prime_x_values.append(f_prime_x)
        
        st.header("Interpolación")
        x_interp = st.text_input("Valores de x a interpolar (separados por comas):")
        x_interp = [float(x.strip()) for x in x_interp.split(",") if x.strip()]
        
        if st.button("Interpolar"):
            interpolated_values = hermite_interpolation(x_values, f_x_values, f_prime_x_values, x_interp)
            
            st.subheader("Resultados:")
            for i, interp_point in enumerate(x_interp):
                st.write(f"Para x = {interp_point}, el valor interpolado es {interpolated_values[i]}")

    if __name__ == "__main__":
        main()