#Proyeecto seminario finanzas streamlit
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from st_aggrid import AgGrid, GridOptionsBuilder


# Funciones auxiliares para optimización
def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negative_sharpe_ratio(weights):
    port_return = np.dot(weights, mean_returns)
    port_volatility = portfolio_volatility(weights)
    return -port_return / port_volatility

def target_return_constraint(weights):
    return np.dot(weights, mean_returns) - 0.10

# Calcular Drawdown
def calculate_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return drawdown.min()

##### Descargar datos desde Yahoo Finance ####
@st.cache_data
def obtener_datos(tickers, start_date, end_date):
    try:
        # Descargar precios ajustados
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return pd.DataFrame()
    
#funcion para convertir los rendiminetos a pesos, multiplicando cada entrada de los activos por el tipo de cambio que estaba en su respectiva fecha el tipo de cambio de Dolares a MXN MXN=X
def data_a_MXN(data,start_date,end_date):
    pesos_mxn = obtener_datos("MXN=X", start_date, end_date)
    data = obtener_datos(all_tickers, start_date, end_date)
    pesos_mxn = pesos_mxn.reindex(data.index, method='ffill') # Rellenar valores faltantes si es necesario
    data = data.multiply(pesos_mxn, axis=0)  # Multiplicación elemento por elemento
    return data


def calcular_black_litterman(data, risk_aversion, P, Q):
    # Calcular rendimientos diarios y matriz de covarianza
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    mean_returns = returns.mean()

    # Pesos de mercado (equitativos)
    market_weights = np.array([1 / data.shape[1]] * data.shape[1])

    # Rendimientos implícitos del mercado (Pi)
    pi = risk_aversion * cov_matrix @ market_weights

    # Parámetro de escala del mercado tau
    tau = 1 / returns.shape[0]
    cov_tau = tau * cov_matrix

    # Omega basado en P y Sigma
    omega = tau * P @ cov_matrix @ P.T

    # Calcular rendimientos ajustados (Black-Litterman)
    inv_cov_tau = np.linalg.inv(cov_tau)
    inv_omega = np.linalg.inv(omega)
    adjusted_returns = np.linalg.inv(inv_cov_tau + P.T @ inv_omega @ P) @ (
        inv_cov_tau @ pi + P.T @ inv_omega @ Q
    )

    # Pesos óptimos
    inv_cov = np.linalg.inv(cov_matrix)
    optimal_weights = inv_cov @ adjusted_returns / (risk_aversion * np.sum(inv_cov @ adjusted_returns))
    normalized_weights = optimal_weights / np.sum(optimal_weights)

    # Resultados finales
    results = pd.DataFrame({
        'Activo': data.columns,
        'Pesos del Mercado': market_weights,
        'Rendimientos Ajustados': adjusted_returns,
        'Pesos Óptimos': normalized_weights
    })

    return results



# Configuración de la página
st.set_page_config(page_title="Análisis individual", layout="wide")
st.sidebar.title("Analizador de Portafolio de Inversión")

####### Barra lateral ########
# Entrada de símbolos de las acciones desde la barra lateral
tickers = st.sidebar.text_input(
    "Ingrese los símbolos de las acciones separados por comas (por ejemplo: ITM,EMB,QQQ,EEM,GLD):",
    "ITM,EMB,QQQ,EEM,GLD"
).split(",")

# Ingreso de pesos desde la barra lateral
pesos = st.sidebar.text_input("Ingrese los pesos correspondientes separados por comas (deben sumar 1):", "0.2,0.2,0.2,0.2,0.2")
pesos = [float(w.strip()) for w in pesos.split(",")]

# Selección del benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}
selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark:", list(benchmark_options.keys()))
benchmark = benchmark_options[selected_benchmark]

# Descargar datos de los tickers y el benchmark
all_tickers = tickers + [benchmark]

# Selección de la ventana de tiempo
end_date = datetime.now()
start_date_options = {
    "1 mes": end_date - timedelta(days=30),
    "3 meses": end_date - timedelta(days=90),
    "6 meses": end_date - timedelta(days=180),
    "1 año": end_date - timedelta(days=365),
    "3 años": end_date - timedelta(days=3*365),
    "5 años": end_date - timedelta(days=5*365),
    "10 años": end_date - timedelta(days=10*365),
    "Seleccionar fechas manualmente": None  # Opción para elegir manualmente
}


###### Menú desplegable para seleccionar la ventana de tiempo ######
selected_window = st.sidebar.selectbox("Seleccione la ventana de tiempo:", list(start_date_options.keys()))

# Configuración de las fechas para la opción de selección manual
end_date = datetime.now().date()  # Convertimos a `datetime.date`
start_date_default = datetime(2010, 1, 1).date()  # Convertimos a `datetime.date`

if selected_window == "Seleccionar fechas manualmente":
    start_date = st.sidebar.date_input(
        "Fecha de inicio:",
        value=start_date_default,
        min_value=datetime(2000, 1, 1).date(),
        max_value=end_date
    )
    end_date = st.sidebar.date_input(
        "Fecha de fin:",
        value=end_date,
        min_value=start_date,
        max_value=datetime.now().date()
    )
else:
    start_date = start_date_options[selected_window]


# Crear pestañas
data = obtener_datos(all_tickers, start_date, end_date)
if not data.empty:
    tab1, tab2, tab3,tab4 = st.tabs(["Análisis de Activos Individuales", "Análisis del Portafolio","Portafolio optimo y backtesting de 2020-2023","Black Litterman"])
    ### Primera Pestaña: Análisis de Activos Individuales ###
    ### Primera Pestaña: Análisis de Activos Individuales ###
    with tab1:
        data = data_a_MXN(obtener_datos(all_tickers, start_date, end_date), start_date, end_date)
        # Calcular precios normalizados para todos los activos
        normalized_prices = data / data.iloc[0] * 100

        #### Barra para elegir el activo ####
        activo_seleccionado = st.selectbox("Seleccione un activo para analizar:", options=tickers)

        ####### Mostrar la descripción del activo seleccionado #######
        try:
            activo_info = yf.Ticker(activo_seleccionado)
            descripcion = activo_info.info.get("longBusinessSummary", "Descripción no disponible")
        except Exception as e:
            descripcion = f"Error al obtener la descripción: {e}"

        # Mostrar descripción en la página
        st.subheader(f"Descripción de {activo_seleccionado}")
        st.write(descripcion)

        ###### Crear el gráfico ajustando el eje Y por ventana de tiempo #####
        fig = go.Figure()

        # Añadir el activo seleccionado
        fig.add_trace(go.Scatter(
            x=normalized_prices.index,
            y=normalized_prices[activo_seleccionado],
            mode='lines',
            name=activo_seleccionado
        ))

        # Añadir el benchmark
        fig.add_trace(go.Scatter(
            x=normalized_prices.index,
            y=normalized_prices[benchmark],
            mode='lines',
            name=selected_benchmark
        ))

        # Configurar escala logarítmica en el eje Y
        fig.update_layout(
            title="Precios Normalizados",
            xaxis_title="Fecha",
            yaxis_title="Precio Normalizado",
            yaxis=dict(type='log')
        )

        # Mostrar el gráfico
        st.plotly_chart(fig)

        ####### Estadísticas del activo seleccionado #######
        returns = data.pct_change().dropna()

        # Estadísticas del activo seleccionado
        stats = {
            "Media Anualizada": returns[activo_seleccionado].mean() * 252,
            "Volatilidad Anualizada": returns[activo_seleccionado].std() * np.sqrt(252),
            "Sesgo": returns[activo_seleccionado].skew(),
            "Exceso de Curtosis": returns[activo_seleccionado].kurt(),
            "VaR (95%)": returns[activo_seleccionado].quantile(0.05),
            "CVaR (95%)": returns[activo_seleccionado][
                returns[activo_seleccionado] < returns[activo_seleccionado].quantile(0.05)
            ].mean(),
            "Sharpe Ratio": (returns[activo_seleccionado].mean() * 252) / (returns[activo_seleccionado].std() * np.sqrt(252)),
            "Sortino Ratio": (returns[activo_seleccionado].mean() * 252) /
                            returns[activo_seleccionado][returns[activo_seleccionado] < 0].std(),
            "Drawdown Máximo": calculate_drawdown(returns[activo_seleccionado])
        }

        # Mostrar métricas en Streamlit
        st.title("Estadísticas del Activo Seleccionado")
        st.subheader(f"Activo: {activo_seleccionado}")

        # Mostrar métricas en columnas
        col1, col2, col3 = st.columns(3)

        col1.metric("Media Anualizada", f"{stats['Media Anualizada']:.2%}")
        col1.metric("Volatilidad", f"{stats['Volatilidad Anualizada']:.2%}")
        col1.metric("Sesgo", f"{stats['Sesgo']:.2f}")

        col2.metric("Curtosis", f"{stats['Exceso de Curtosis']:.2f}")
        col2.metric("VaR (95%)", f"{stats['VaR (95%)']:.2%}")
        col2.metric("CVaR (95%)", f"{stats['CVaR (95%)']:.2%}")

        col3.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
        col3.metric("Sortino Ratio", f"{stats['Sortino Ratio']:.2f}")
        col3.metric("Drawdown Máximo", f"{stats['Drawdown Máximo']:.2%}")

        ### Segunda Pestaña: Análisis del Portafolio ###
    with tab2:
        st.title("Análisis del Portafolio")

        if not data.empty:
            # Calcular los retornos diarios del portafolio y benchmark
            start_date_fixed = datetime(2010, 1, 1)
            end_date_fixed = datetime(2020, 12, 31)
            data = data_a_MXN(obtener_datos(tickers, start_date, end_date), start_date, end_date)
            returns = data.pct_change().dropna()
            portfolio_returns = (returns[tickers] * pesos).sum(axis=1)
            portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()

            # Gráfico de rendimientos acumulados
            st.subheader("Rendimientos Acumulados del Portafolio y Benchmark")
            benchmark_cumulative_returns = (1 + returns[benchmark]).cumprod()
            cumulative_data = pd.DataFrame({
                "Portafolio": portfolio_cumulative_returns,
                selected_benchmark: benchmark_cumulative_returns
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_data.index,
                y=cumulative_data["Portafolio"],
                mode='lines',
                name="Portafolio"
            ))
            fig.add_trace(go.Scatter(
                x=cumulative_data.index,
                y=cumulative_data[selected_benchmark],
                mode='lines',
                name=selected_benchmark
            ))
            fig.update_layout(
                title="Rendimientos Acumulados (Escala Logarítmica)",
                xaxis_title="Fecha",
                yaxis_title="Rendimiento Acumulado",
                yaxis=dict(type="log")
            )
            st.plotly_chart(fig)

            # Estadísticas del portafolio
            st.subheader("Estadísticas del Portafolio")
            portfolio_stats = pd.DataFrame(index=["Portafolio"])
            portfolio_stats["Rendimiento Promedio"] = [portfolio_returns.mean() * 252]
            portfolio_stats["Volatilidad"] = [portfolio_returns.std() * np.sqrt(252)]
            portfolio_stats["Sharpe Ratio"] = [
                portfolio_stats["Rendimiento Promedio"].iloc[0] / portfolio_stats["Volatilidad"].iloc[0]
            ]
            col1, col2, col3 = st.columns(3)
            col1.metric("Rendimiento Promedio", f"{portfolio_stats['Rendimiento Promedio'].iloc[0]:.2%}")
            col2.metric("Volatilidad", f"{portfolio_stats['Volatilidad'].iloc[0]:.2%}")
            col3.metric("Sharpe Ratio", f"{portfolio_stats['Sharpe Ratio'].iloc[0]:.2f}")

            # Cálculo de Pesos Óptimos
            st.subheader("Cálculo de Pesos Óptimos")

            # Filtrar retornos para el período seleccionado
            returns_opt = returns.loc[start_date:end_date]

            # Verificar si hay datos suficientes
            if not returns_opt.empty:
                # Filtrar activos con datos disponibles en el rango de fechas
                available_assets = returns_opt.columns
                filtered_tickers = [ticker for ticker in tickers if ticker in available_assets]

                # Recalcular los rendimientos medios y la matriz de covarianza
                mean_returns = returns_opt[filtered_tickers].mean() * 252
                cov_matrix = returns_opt[filtered_tickers].cov() * 252
                num_assets = len(filtered_tickers)

                # Definir restricciones y límites
                bounds = [(0, 1) for _ in range(num_assets)]
                constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

                # Optimizar para diferentes objetivos
                min_vol_port = minimize(portfolio_volatility, [1/num_assets]*num_assets, bounds=bounds, constraints=constraints)
                max_sharpe_port = minimize(negative_sharpe_ratio, [1/num_assets]*num_assets, bounds=bounds, constraints=constraints)
                constraints_with_target = constraints + [{'type': 'eq', 'fun': target_return_constraint}]
                target_return_port = minimize(portfolio_volatility, [1/num_assets]*num_assets, bounds=bounds, constraints=constraints_with_target)

                # Crear DataFrame con los pesos óptimos
                portfolios = pd.DataFrame({
                    "Portafolio": ["Mínima Volatilidad", "Máximo Sharpe", "Rendimiento Objetivo"],
                    "Pesos": [
                        min_vol_port.x,
                        max_sharpe_port.x,
                        target_return_port.x
                    ]
                })

                
                # Slider para seleccionar la estrategia óptima
                st.subheader("Seleccione una Estrategia Óptima")
                estrategia_seleccionada = st.radio(
                    "Opciones de Optimización:",
                    ["Mínima Volatilidad", "Máximo Sharpe", "Rendimiento Objetivo"]
                )

                # Calcular los pesos según la estrategia seleccionada
                pesos_seleccionados = None

                if estrategia_seleccionada == "Mínima Volatilidad":
                    pesos_seleccionados = min_vol_port.x
                elif estrategia_seleccionada == "Máximo Sharpe":
                    pesos_seleccionados = max_sharpe_port.x
                elif estrategia_seleccionada == "Rendimiento Objetivo":
                    pesos_seleccionados = target_return_port.x

                # Configurar dos columnas para mostrar la gráfica de dona y la tabla
                col1, col2 = st.columns(2)

                # Tabla 
                with col1:
                    st.subheader("Pesos Óptimos")
                    st.markdown("### Distribución de Pesos:")
                    for activo, peso in zip(filtered_tickers, pesos_seleccionados):
                        st.markdown(f"- **{activo}**: {round(peso * 100, 2)}%")
                
                # Gráfica de dona
                with col2:
                    st.subheader(f"Distribución de Pesos Seleccionados ({estrategia_seleccionada})")
                    fig_dona = px.pie(
                        names=filtered_tickers,
                        values=pesos_seleccionados,
                        hole=0.5  # Para hacer una dona
                    )
                    fig_dona.update_traces(textinfo='percent+label', textfont_size=12)
                    st.plotly_chart(fig_dona, use_container_width=True)

                # Crear una nueva gráfica con los pesos seleccionados
                st.subheader(f"Rendimientos del Portafolio usando Pesos de {estrategia_seleccionada}")

                # Calcular los rendimientos del portafolio con los pesos seleccionados
                portfolio_returns_estrategia = (returns[filtered_tickers] * pesos_seleccionados).sum(axis=1)
                portfolio_cumulative_estrategia = (1 + portfolio_returns_estrategia).cumprod()

                # Crear el DataFrame para la gráfica
                cumulative_data_estrategia = pd.DataFrame({
                    "Portafolio": portfolio_cumulative_estrategia,
                    selected_benchmark: benchmark_cumulative_returns
                })
                # Calcular métricas del portafolio ajustado

                adjusted_portfolio_stats = pd.DataFrame(index=["Portafolio"])
                adjusted_portfolio_stats["Rendimiento Promedio"] = [portfolio_returns_estrategia.mean() * 252]
                adjusted_portfolio_stats["Volatilidad"] = [portfolio_returns_estrategia.std() * np.sqrt(252)]
                adjusted_portfolio_stats["Sharpe Ratio"] = [
                    adjusted_portfolio_stats["Rendimiento Promedio"].iloc[0] / adjusted_portfolio_stats["Volatilidad"].iloc[0]
                ]
                # Mostrar las métricas
                st.subheader("Estadísticas del Portafolio Ajustado")
                col4, col5, col6 = st.columns(3)
                col4.metric("Rendimiento Promedio", f"{adjusted_portfolio_stats['Rendimiento Promedio'].iloc[0]:.2%}")
                col5.metric("Volatilidad", f"{adjusted_portfolio_stats['Volatilidad'].iloc[0]:.2%}")
                col6.metric("Sharpe Ratio", f"{adjusted_portfolio_stats['Sharpe Ratio'].iloc[0]:.2f}")
                # Graficar el portafolio y el benchmark
                fig_estrategia = go.Figure()
                fig_estrategia.add_trace(go.Scatter(
                    x=cumulative_data_estrategia.index,
                    y=cumulative_data_estrategia["Portafolio"],
                    mode='lines',
                    name="Portafolio"
                ))
                fig_estrategia.add_trace(go.Scatter(
                    x=cumulative_data_estrategia.index,
                    y=cumulative_data_estrategia[selected_benchmark],
                    mode='lines',
                    name=selected_benchmark
                ))
                fig_estrategia.update_layout(
                    title=f"Rendimientos Acumulados con Pesos de {estrategia_seleccionada} (Escala Logarítmica)",
                    xaxis_title="Fecha",
                    yaxis_title="Rendimiento Acumulado",
                    yaxis=dict(type="log")
                )
                st.plotly_chart(fig_estrategia)  
            else:
                st.error("No hay suficientes datos en el rango de fechas seleccionado.")
        else:
            st.error("No se pudieron obtener datos de los activos o el benchmark.")
            
    ##### termina hoja 2 ####

    with tab3:# Título de la pestaña 3
        st.title("Portafolio Óptimo al 10% y Backtesting")

        # Configurar las fechas fijas para el análisis
        start_date_fixed = datetime(2010, 1, 1)
        end_date_fixed = datetime(2020, 12, 31)

        ##### Descargar datos para el rango fijo de fechas, incluyendo el benchmark ####
        data_fixed = obtener_datos(tickers + ["^GSPC"], start_date_fixed, end_date_fixed)
        data_fixed = data_a_MXN(data_fixed,start_date_fixed, end_date_fixed)
        

        if not data_fixed.empty:
            # Calcular los retornos diarios y realizar la optimización
            returns_fixed = data_fixed.pct_change().dropna()
            mean_returns_fixed = returns_fixed[tickers].mean() * 252
            cov_matrix_fixed = returns_fixed[tickers].cov() * 252
            num_assets_fixed = len(tickers)

            # Restricciones y límites
            bounds = [(0, 1) for _ in range(num_assets_fixed)]
            constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
            constraints_with_target = constraints + [{'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns_fixed) - 0.10}]

            # Optimización para el portafolio objetivo (10% de rendimiento anual)
            optimized_portfolio_fixed = minimize(
                portfolio_volatility,
                [1 / num_assets_fixed] * num_assets_fixed,
                bounds=bounds,
                constraints=constraints_with_target
            )

            # Obtener los pesos óptimos calculados
            pesos_optimos_fixed = optimized_portfolio_fixed.x

            # Calcular los rendimientos acumulados de 2010 a 2020 con los pesos óptimos
            portfolio_returns_fixed = (returns_fixed[tickers] * pesos_optimos_fixed).sum(axis=1)
            portfolio_cumulative_fixed = (1 + portfolio_returns_fixed).cumprod()
            
            

            # Calcular los rendimientos acumulados del S&P 500
            benchmark_returns_fixed = returns_fixed["^GSPC"]
            benchmark_cumulative_fixed = (1 + benchmark_returns_fixed).cumprod()

            # Descargar datos para el período 2021-2023
            start_date_new2 = datetime(2021, 1, 1)
            end_date_new2 = datetime(2023, 12, 31)
            data_new = obtener_datos(tickers + ["^GSPC"], start_date_new2, end_date_new2)
            data_new=data_a_MXN(data_new,start_date_new2,end_date_new2)


            if not data_new.empty:
                # Calcular los rendimientos acumulados de 2021 a 2023 usando los mismos pesos óptimos
                returns_new = data_new.pct_change().dropna()
                portfolio_returns_new = (returns_new[tickers] * pesos_optimos_fixed).sum(axis=1)
                portfolio_cumulative_new = (1 + portfolio_returns_new).cumprod()

                adjusted_portfolio_stats["Rendimiento Promedio"] = [portfolio_returns_new.mean() * 252]
                adjusted_portfolio_stats["Volatilidad"] = [portfolio_returns_new.std() * np.sqrt(252)]
                adjusted_portfolio_stats["Sharpe Ratio"] = [adjusted_portfolio_stats["Rendimiento Promedio"].iloc[0] / adjusted_portfolio_stats["Volatilidad"].iloc[0]]
                
                # Métricas
                metrics_fixed = {
                    "Rendimiento Anualizado": portfolio_returns_fixed.mean() * 252,
                    "Volatilidad Anualizada": portfolio_returns_fixed.std() * np.sqrt(252),
                    "Sharpe Ratio": (portfolio_returns_fixed.mean() * 252) / (portfolio_returns_fixed.std() * np.sqrt(252)),
                    "Drawdown Máximo": (portfolio_cumulative_fixed / portfolio_cumulative_fixed.cummax() - 1).min(),
                    "Media": portfolio_returns_fixed.mean(),
                    "Sesgo": portfolio_returns_fixed.skew(),
                    "Exceso de Curtosis": portfolio_returns_fixed.kurt(),
                    "VaR (95%)": portfolio_returns_fixed.quantile(0.05),  # Percentil 5%
                    "CVaR (95%)": portfolio_returns_fixed[portfolio_returns_fixed <= portfolio_returns_fixed.quantile(0.05)].mean(),
                    "Sortino Ratio": (portfolio_returns_fixed.mean() * 252) /
                                    portfolio_returns_fixed[portfolio_returns_fixed < 0].std() * np.sqrt(252) if portfolio_returns_fixed[portfolio_returns_fixed < 0].std() != 0 else np.nan
                }
                # Mostrar métricas 2020-2023
                st.subheader("Estadísticas del Portafolio (2010-2020)")
                col1, col2, col3 = st.columns(3)

                
                col1.metric("Rendimiento Anualizado", f"{metrics_fixed['Rendimiento Anualizado']:.2%}")
                col1.metric("Volatilidad Anualizada", f"{metrics_fixed['Volatilidad Anualizada']:.2%}")
                col1.metric("Sharpe Ratio", f"{metrics_fixed['Sharpe Ratio']:.2f}")
                col1.metric("Sortino Ratio", f"{metrics_fixed['Sortino Ratio']:.2f}" if not pd.isna(metrics_fixed['Sortino Ratio']) else "N/A")

                col2.metric("Media", f"{metrics_fixed['Media']:.5f}")
                col2.metric("Sesgo", f"{metrics_fixed['Sesgo']:.2f}")
                col2.metric("Exceso de Curtosis", f"{metrics_fixed['Exceso de Curtosis']:.2f}")

                col3.metric("Drawdown Máximo", f"{metrics_fixed['Drawdown Máximo']:.2%}")
                col3.metric("VaR (95%)", f"{metrics_fixed['VaR (95%)']:.2%}")
                col3.metric("CVaR (95%)", f"{metrics_fixed['CVaR (95%)']:.2%}")


                # Calcular los rendimientos acumulados del S&P 500
                benchmark_returns_new = returns_new["^GSPC"]
                benchmark_cumulative_new = (1 + benchmark_returns_new).cumprod()

                # Gráfica 1: Rendimientos acumulados de 2010 a 2020
                st.subheader("Rendimientos Acumulados (2010-2020)")
                fig_fixed = go.Figure()
                fig_fixed.add_trace(go.Scatter(
                    x=portfolio_cumulative_fixed.index,
                    y=portfolio_cumulative_fixed,
                    mode='lines',
                    name="Portafolio (2010-2020)"
                ))
                fig_fixed.add_trace(go.Scatter(
                    x=benchmark_cumulative_fixed.index,
                    y=benchmark_cumulative_fixed,
                    mode='lines',
                    name="S&P 500 (2010-2020)"
                ))
                fig_fixed.update_layout(
                    title="Portafolio Óptimo vs S&P 500 (2010-2020)",
                    xaxis_title="Fecha",
                    yaxis_title="Rendimiento Acumulado"
                )
                st.plotly_chart(fig_fixed)

                # Cálculo de métricas
                metrics_new = {
                    "Rendimiento Anualizado": portfolio_returns_new.mean() * 252,
                    "Volatilidad Anualizada": portfolio_returns_new.std() * np.sqrt(252),
                    "Sharpe Ratio": (portfolio_returns_new.mean() * 252) / (portfolio_returns_new.std() * np.sqrt(252)),
                    "Drawdown Máximo": ((1 + portfolio_returns_new).cumprod() / (1 + portfolio_returns_new).cumprod().cummax() - 1).min(),
                    "Media": portfolio_returns_new.mean(),
                    "Sesgo": portfolio_returns_new.skew(),
                    "Exceso de Curtosis": portfolio_returns_new.kurt(),
                    "VaR (95%)": portfolio_returns_new.quantile(0.05),  # Percentil 5%
                    "CVaR (95%)": portfolio_returns_new[portfolio_returns_new <= portfolio_returns_new.quantile(0.05)].mean(),
                    "Sortino Ratio": (portfolio_returns_new.mean() * 252) /
                                    portfolio_returns_new[portfolio_returns_new < 0].std() * np.sqrt(252) if portfolio_returns_new[portfolio_returns_new < 0].std() != 0 else np.nan
                }

                # Metricas
                st.subheader("Estadísticas del Portafolio (2021-2023)")
                col1, col2, col3 = st.columns(3)

                col1.metric("Rendimiento Anualizado", f"{metrics_new['Rendimiento Anualizado']:.2%}")
                col1.metric("Volatilidad Anualizada", f"{metrics_new['Volatilidad Anualizada']:.2%}")
                col1.metric("Sharpe Ratio", f"{metrics_new['Sharpe Ratio']:.2f}")
                col1.metric("Sortino Ratio", f"{metrics_new['Sortino Ratio']:.2f}" if not pd.isna(metrics_new['Sortino Ratio']) else "N/A")

                col2.metric("Media", f"{metrics_new['Media']:.5f}")
                col2.metric("Sesgo", f"{metrics_new['Sesgo']:.2f}")
                col2.metric("Exceso de Curtosis", f"{metrics_new['Exceso de Curtosis']:.2f}")

                col3.metric("Drawdown Máximo", f"{metrics_new['Drawdown Máximo']:.2%}")
                col3.metric("VaR (95%)", f"{metrics_new['VaR (95%)']:.2%}")
                col3.metric("CVaR (95%)", f"{metrics_new['CVaR (95%)']:.2%}")

                # Gráfica 2: Rendimientos acumulados de 2021 a 2023
                st.subheader("Rendimientos Acumulados (2021-2023) usando Pesos de 2010-2020")
                fig_new = go.Figure()
                fig_new.add_trace(go.Scatter(
                    x=portfolio_cumulative_new.index,
                    y=portfolio_cumulative_new,
                    mode='lines',
                    name="Portafolio (2021-2023)"
                ))
                fig_new.add_trace(go.Scatter(
                    x=benchmark_cumulative_new.index,
                    y=benchmark_cumulative_new,
                    mode='lines',
                    name="S&P 500 (2021-2023)"
                ))
                fig_new.update_layout(
                    title="Portafolio Aplicado vs S&P 500 (2021-2023)",
                    xaxis_title="Fecha",
                    yaxis_title="Rendimiento Acumulado"
                )
                st.plotly_chart(fig_new)
            else:
                st.error("No se pudieron obtener datos de 2021 a 2023 para los activos seleccionados.")
        else:
            st.error("No se pudieron obtener datos de 2010 a 2020 para los activos seleccionados.")
    ### fin tab 3 ####

    with tab4:
        st.title("Modelo Black-Litterman")

        # Descargar datos ajustados a pesos
        data = data_a_MXN(obtener_datos(tickers, start_date, end_date), start_date, end_date)

        # Excluir el benchmark de los activos
        activos_sin_benchmark = [col for col in data.columns if col != benchmark]
        data = data[activos_sin_benchmark]  # Filtrar columnas del DataFrame
        num_activos = len(data.columns)

        # Configuración de parámetros
        st.markdown("### Configuración de Parámetros")
        col1, col2 = st.columns(2)
        with col1:
            risk_aversion = st.number_input("Aversión al riesgo (λ):", min_value=0.1, max_value=5.0, value=2.4, step=0.1)
        with col2:
            num_opiniones = st.number_input("Número de Opiniones:", min_value=1, max_value=5, value=2, step=1)

        # Crear tabla de configuración de opiniones
        st.markdown("### Tabla asociada a la matriz P y Q")
        st.text("En la columna peso 1 si el rendimiento que se espera positivo, -1 en otro caso, 0 si no evaluamos ese activo.")
        opinion_data = []   
        for i in range(num_opiniones):
            for activo in data.columns:
                opinion_data.append({
                    "Opinión": f"Opinión {i + 1}",
                    "Activo": activo,
                    "Peso": 0.0,  # Peso inicial
                    "Rendimiento Esperado (%)": 1.0 if i == 0 else 0.5  # Rendimiento inicial
                })

        # Configuración de la tabla editable
        df_opinion = pd.DataFrame(opinion_data)
        gb = GridOptionsBuilder.from_dataframe(df_opinion)
        gb.configure_default_column(editable=True)  # Hacer todas las celdas editables
        gb.configure_column("Opinión", editable=False)  # No permitir edición en la columna "Opinión"
        gb.configure_column("Activo", editable=False)  # No permitir edición en la columna "Activo"
        gb.configure_column("Peso", type=["numericColumn"], precision=2, editable=True)
        gb.configure_column("Rendimiento Esperado (%)", type=["numericColumn"], precision=2, editable=True)
        grid_options = gb.build()
        grid_response = AgGrid(
            df_opinion,
            gridOptions=grid_options,
            editable=True,
            height=400,
            width="100%",
            theme="streamlit"  # Tema de la tabla
        )

        # Obtener los datos editados de la tabla
        edited_data = pd.DataFrame(grid_response["data"])

        # Procesar matriz P y vector Q
        P = np.zeros((num_opiniones, num_activos))
        Q = []  # Inicializar Q como lista

        for i in range(num_opiniones):
            opinion_subset = edited_data[edited_data["Opinión"] == f"Opinión {i + 1}"]
            activos = opinion_subset["Activo"].values
            pesos = opinion_subset["Peso"].values

            # Validar y normalizar los pesos
            if np.sum(pesos) > 0:
                pesos = pesos / np.sum(pesos)  # Normalizar para que sumen 1
            else:
                st.error(f"Los pesos de la Opinión {i + 1} no pueden ser cero. Revisa los valores.")
                continue

            rendimiento = opinion_subset["Rendimiento Esperado (%)"].mean() / 100  # Convertir a proporción

            # Actualizar la matriz P
            for j, activo in enumerate(activos):
                idx = data.columns.tolist().index(activo)
                P[i, idx] = pesos[j]

            # Agregar rendimiento a Q como una lista
            Q.append(rendimiento)

        # Convertir Q a un numpy.ndarray después de construirlo completamente
        Q = np.array(Q)

        # Botón para calcular Black-Litterman
        if st.button("Calcular Black-Litterman"):
            try:
                resultados = calcular_black_litterman(data, risk_aversion, P, Q)

                # Mostrar resultados en formato de texto
                st.write("### Resultados del Modelo Black-Litterman")
                for i, row in resultados.iterrows():
                    st.markdown(
                        f"""
                        **Activo:** {row['Activo']}  
                        - **Pesos del Mercado:** {row['Pesos del Mercado'] * 100:.2f}%  
                        - **Rendimientos Ajustados:** {row['Rendimientos Ajustados'] * 100:.2f}%  
                        - **Pesos Óptimos:** {row['Pesos Óptimos'] * 100:.2f}%
                        """
                    )

                # Gráfico de comparación de pesos (Barras)
                st.write("### Comparación de Pesos: Mercado vs. Black-Litterman")
                fig, ax = plt.subplots(figsize=(6, 4))  # Cambia figsize a un tamaño más compacto
                resultados.set_index('Activo')[['Pesos del Mercado', 'Pesos Óptimos']].plot(
                    kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e']
                )
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Línea en 0
                ax.set_ylabel('Pesos (%)')
                ax.set_title('Pesos Óptimos vs Pesos de Mercado', fontsize=12)  # Tamaño de fuente más pequeño
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error en el cálculo del modelo Black-Litterman: {e}")
else:
    st.error("No se pudieron obtener datos de los activos o el benchmark.")

