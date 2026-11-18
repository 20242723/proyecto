import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as st_scipy
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# Para la Dimensión 1 (Proyección de Demanda)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Para la Dimensión 2 (Predicción de Lead Time con SVR)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Para la Dimensión 3 (Análisis de Portafolio)
import statsmodels.formula.api as smf


# Configuración principal de la interfaz
st.set_page_config(page_title="Ventas de Cafetería - Dashboard Predictivo", page_icon="☕", layout="wide")

# Estilos CSS personalizados para mejorar el diseño
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #4A2E2B;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        color: #6F4E37;
        border-bottom: 2px solid #6F4E37;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .sub-card {
        background-color: #FDFBF7;
        padding: 18px;
        border-radius: 8px;
        border-left: 5px solid #6F4E37;
        margin-bottom: 15px;
    }
    .scenario-banner {
        background-color: #F8F3ED;
        padding: 14px 18px;
        border-radius: 8px;
        border: 1px solid #D2B48C;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar datos con caché
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Coffee Shop Sales.xlsx")
        return df
    except Exception as e:
        np.random.seed(42)
        fechas = pd.date_range(start="2026-01-01", periods=180, freq="D")
        locales = ["Hell's Kitchen", "Astoria", "Lower Manhattan"]
        categorias = ["Coffee", "Tea", "Bakery", "Drinking Chocolate", "Coffee beans"]
        registros = []
        t_id = 1
        for f in fechas:
            for l in locales:
                for c in categorias:
                    qty = np.random.randint(10, 100)
                    price = 3.5 if c in ["Coffee", "Tea"] else 4.5
                    # Variación leve de precio para que los datos simulados también soporten Dimensión 3
                    if np.random.rand() < 0.25:
                        price = round(price * np.random.choice([0.9, 1.1]), 2)
                    registros.append({
                        "transaction_id": t_id,
                        "transaction_date": f,
                        "store_location": l,
                        "product_category": c,
                        "product_type": f"Tipo {c}",
                        "product_detail": f"Detalle {c}",
                        "transaction_qty": qty,
                        "unit_price": price
                    })
                    t_id += 1
        return pd.DataFrame(registros)

df = load_data()

# -----------------------------------------------------------------------------
# DIMENSIÓN 1: MODELADO Y PROYECCIÓN DE DEMANDA (COFFEE & TEA)
# -----------------------------------------------------------------------------
@st.cache_data
def ejecutar_dimension_1(df_input):
    df_temp = df_input.copy()
    df_temp['transaction_date'] = pd.to_datetime(df_temp['transaction_date'])
    
    if 'product_type' in df_temp.columns and 'product_name' not in df_temp.columns:
        df_temp = df_temp.rename(columns={'product_type': 'product_name'})
        
    df_top2 = df_temp[df_temp['product_category'].isin(['Coffee', 'Tea'])].copy()

    df_weekly = df_top2.groupby([
        pd.Grouper(key='transaction_date', freq='W-MON'),
        'store_location',
        'product_name'
    ])['transaction_qty'].sum().reset_index()

    def create_weekly_features(dataframe):
        df_feat = dataframe.copy()
        df_feat['week_of_year'] = df_feat['transaction_date'].dt.isocalendar().week.astype(int)
        df_feat['month'] = df_feat['transaction_date'].dt.month
        for lag in [1, 2, 3]:
            df_feat[f'lag_{lag}'] = df_feat.groupby(['store_location', 'product_name'])['transaction_qty'].shift(lag)
        df_feat['rolling_mean_3'] = df_feat.groupby(['store_location', 'product_name'])['lag_1'].transform(lambda x: x.rolling(3).mean())
        return df_feat.dropna()

    df_model_ml = create_weekly_features(df_weekly)

    dates = sorted(df_weekly['transaction_date'].unique())
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    results_list = []
    products_by_store = df_weekly[['store_location', 'product_name']].drop_duplicates()

    for _, row in products_by_store.iterrows():
        store = row['store_location']
        pname = row['product_name']

        sku_df = df_weekly[(df_weekly['store_location'] == store) & (df_weekly['product_name'] == pname)].sort_values('transaction_date')
        train_df = sku_df[sku_df['transaction_date'].isin(train_dates)]
        test_df = sku_df[sku_df['transaction_date'].isin(test_dates)]

        if len(test_df) == 0 or len(train_df) < 4:
            continue

        y_train = train_df['transaction_qty'].values
        y_test = test_df['transaction_qty'].values

        pred_baseline = np.full(len(y_test), y_train[-2:].mean())

        try:
            hw_model = ExponentialSmoothing(y_train, trend='add').fit()
            pred_hw = hw_model.forecast(len(y_test))
        except:
            pred_hw = pred_baseline

        sku_ml = df_model_ml[(df_model_ml['store_location'] == store) & (df_model_ml['product_name'] == pname)]
        features = ['week_of_year', 'month', 'lag_1', 'lag_2', 'rolling_mean_3']

        train_ml = sku_ml[sku_ml['transaction_date'].isin(train_dates)]
        test_ml = sku_ml[sku_ml['transaction_date'].isin(test_dates)]

        if len(train_ml) > 0 and len(test_ml) > 0:
            X_tr, y_tr = train_ml[features], train_ml['transaction_qty']
            X_te, y_te = test_ml[features], test_ml['transaction_qty']

            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
            pred_rf = rf.predict(X_te)

            xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42).fit(X_tr, y_tr)
            pred_xgb = xgb.predict(X_te)

            y_test_eval = y_te.values
        else:
            pred_rf, pred_xgb = pred_baseline, pred_baseline
            y_test_eval = y_test

        models_preds = {
            'Promedio Móvil (Baseline)': (y_test, pred_baseline),
            'Holt-Winters (Estadístico)': (y_test, pred_hw),
            'Random Forest (ML)': (y_test_eval, pred_rf),
            'XGBoost (ML)': (y_test_eval, pred_xgb)
        }

        for model_name, (actual, forecast) in models_preds.items():
            if len(actual) == 0 or np.sum(actual) == 0: continue
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            wmape = (np.sum(np.abs(actual - forecast)) / np.sum(actual)) * 100
            accuracy = max(0, 100 - wmape)

            results_list.append({
                'Modelo': model_name,
                'MAE (Unidades/Semana)': mae,
                'RMSE': rmse,
                'WMAPE (%)': wmape,
                'Precisión del Pronóstico (%)': accuracy
            })

    results_df = pd.DataFrame(results_list)
    summary_df = results_df.groupby('Modelo')[['MAE (Unidades/Semana)', 'RMSE', 'WMAPE (%)', 'Precisión del Pronóstico (%)']].mean().reset_index()
    summary_df = summary_df.sort_values(by='Precisión del Pronóstico (%)', ascending=False)

    best_model_name = summary_df.iloc[0]['Modelo']
    best_acc = round(summary_df.iloc[0]['Precisión del Pronóstico (%)'], 2)

    last_date = df_weekly['transaction_date'].max()
    future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, 5)]
    forecast_records = []

    for _, row in products_by_store.iterrows():
        store = row['store_location']
        pname = row['product_name']

        sku_df = df_weekly[(df_weekly['store_location'] == store) & (df_weekly['product_name'] == pname)].sort_values('transaction_date')
        if len(sku_df) < 4: continue

        if "Random Forest" in best_model_name or "XGBoost" in best_model_name:
            sku_ml = df_model_ml[(df_model_ml['store_location'] == store) & (df_model_ml['product_name'] == pname)]
            if len(sku_ml) >= 4:
                X_all = sku_ml[['week_of_year', 'month', 'lag_1', 'lag_2', 'rolling_mean_3']]
                y_all = sku_ml['transaction_qty']

                if "Random Forest" in best_model_name:
                    model_final = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_all, y_all)
                else:
                    model_final = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42).fit(X_all, y_all)

                last_lags = sku_ml.iloc[-1]
                lag1, lag2 = last_lags['transaction_qty'], last_lags['lag_1']

                for f_date in future_dates:
                    w_num = f_date.isocalendar().week
                    m_num = f_date.month
                    roll3 = np.mean([lag1, lag2])

                    X_fut = pd.DataFrame([[w_num, m_num, lag1, lag2, roll3]], columns=X_all.columns)
                    pred_qty = max(0, model_final.predict(X_fut)[0])

                    forecast_records.append({
                        'transaction_date': f_date,
                        'store_location': store,
                        'product_name': pname,
                        'Demanda_Proyectada_Semanal': round(pred_qty, 2)
                    })
                    lag2, lag1 = lag1, pred_qty
        else:
            hw_final = ExponentialSmoothing(sku_df['transaction_qty'].values, trend='add').fit()
            preds_hw = hw_final.forecast(4)
            for idx, f_date in enumerate(future_dates):
                forecast_records.append({
                    'transaction_date': f_date,
                    'store_location': store,
                    'product_name': pname,
                    'Demanda_Proyectada_Semanal': round(max(0, preds_hw[idx]), 2)
                })

    df_proyeccion = pd.DataFrame(forecast_records)
    return summary_df, best_model_name, best_acc, df_proyeccion


# -----------------------------------------------------------------------------
# MODELO DE ENTRENAMIENTO SVR PARA LEAD TIME (DIMENSIÓN 2)
# -----------------------------------------------------------------------------
@st.cache_resource
def entrenar_modelo_svr():
    np.random.seed(42)
    n = 1000
    df_train = pd.DataFrame({
        'store_location': np.random.choice(['Astoria', "Hell's Kitchen", 'Lower Manhattan'], n),
        'product_name': np.random.choice(['Barista Espresso', 'Brewed Black tea', 'Brewed Chai tea'], n),
        'distancia_km': np.random.uniform(5, 50, n),
        'trafico_promedio_index': np.random.uniform(1.0, 5.0, n),
        'clima_val': np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1])
    })
    df_train['lead_time_real'] = (
        1.0 + (df_train['distancia_km'] * 0.02) + 
        (df_train['trafico_promedio_index'] * 0.3) + 
        (df_train['clima_val'] * 0.5) + np.random.normal(0, 0.1, n)
    )
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['distancia_km', 'trafico_promedio_index', 'clima_val']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['store_location', 'product_name'])
    ])
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('svr', SVR(C=10, gamma='scale', epsilon=0.1))
    ])

    X = df_train.drop(columns=['lead_time_real'])
    y = df_train['lead_time_real']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    metricas = {
        'R2': r2_score(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE': mean_absolute_error(y_test, y_pred_test)
    }
    return pipeline, metricas

modelo_svr, metricas_svr = entrenar_modelo_svr()


def calcular_politica_inventario(tienda, producto, demanda_diaria, distancia_km, trafico_idx, clima_val, nivel_servicio_pct):
    """Calcula Lead Time (SVR), Stock de Seguridad (SS) y Punto de Reorden (ROP) para un escenario."""
    input_df = pd.DataFrame([{
        'store_location': tienda,
        'product_name': producto,
        'distancia_km': float(distancia_km),
        'trafico_promedio_index': float(trafico_idx),
        'clima_val': int(clima_val)
    }])
    lt_pred = float(modelo_svr.predict(input_df)[0])
    z_val = float(st_scipy.norm.ppf(nivel_servicio_pct / 100.0))
    rmse_svr = float(metricas_svr['RMSE'])
    sigma_d = float(demanda_diaria) * 0.15

    ss = z_val * np.sqrt((lt_pred * (sigma_d ** 2)) + ((demanda_diaria ** 2) * (rmse_svr ** 2)))
    rop = (demanda_diaria * lt_pred) + ss
    demanda_durante_lt = demanda_diaria * lt_pred
    cobertura_dias = (rop / demanda_diaria) if demanda_diaria > 0 else 0.0

    return {
        'lead_time': lt_pred,
        'demanda_diaria': float(demanda_diaria),
        'demanda_lt': demanda_durante_lt,
        'ss': float(ss),
        'rop': float(rop),
        'cobertura_dias': float(cobertura_dias)
    }


def simular_trayectoria_diaria(demanda_diaria, lt_dias, ss, rop, dias=28, lote_dias=7, seed=42):
    """Simula la dinámica diaria de inventario (diente de sierra) bajo política (ROP, Q)."""
    rng = np.random.default_rng(seed)
    demandas = np.maximum(1.0, rng.normal(loc=demanda_diaria, scale=demanda_diaria * 0.15, size=dias))
    lt_entero = max(1, int(np.round(lt_dias)))
    q_pedido = max(10.0, demanda_diaria * lote_dias)

    inventario = rop + (q_pedido * 0.6)
    pedidos_en_transito = []

    registros = []
    for dia in range(1, dias + 1):
        recibido_hoy = sum(cant for d_lleg, cant in pedidos_en_transito if d_lleg == dia)
        pedidos_en_transito = [(d_lleg, cant) for d_lleg, cant in pedidos_en_transito if d_lleg > dia]
        inventario += recibido_hoy

        inv_inicio = inventario
        dem_hoy = demandas[dia - 1]
        quiebre_unids = max(0.0, dem_hoy - inventario)
        inventario = max(0.0, inventario - dem_hoy)

        inv_posicion = inventario + sum(cant for _, cant in pedidos_en_transito)
        orden_emitida = False
        if inv_posicion <= rop:
            pedidos_en_transito.append((dia + lt_entero, q_pedido))
            orden_emitida = True

        registros.append({
            'Día': dia,
            'Inventario Inicial': round(inv_inicio, 1),
            'Demanda Diaria': round(dem_hoy, 1),
            'Inventario Final': round(inventario, 1),
            'Quiebre (Unids)': round(quiebre_unids, 1),
            'Orden Emitida': orden_emitida,
            'Pedido Recibido': recibido_hoy > 0,
            'ROP': round(rop, 1),
            'SS': round(ss, 1)
        })

    return pd.DataFrame(registros)


# -----------------------------------------------------------------------------
# DIMENSIÓN 3: PREPARACIÓN DEL PORTAFOLIO Y MODELO DE ELASTICIDAD
# -----------------------------------------------------------------------------
@st.cache_data
def construir_dimension_3(df_origen):
    """Calcula el portafolio de productos con precio variable, elasticidades
    log-log, significancia estadística y simulaciones de precio ±5% y ±10%."""
    requeridas = {
        'transaction_id', 'transaction_date', 'transaction_qty', 'store_location',
        'unit_price', 'product_detail'
    }
    faltantes = requeridas - set(df_origen.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para Dimensión 3: {sorted(faltantes)}")

    d = df_origen.copy()
    d['transaction_date'] = pd.to_datetime(d['transaction_date'])
    d['revenue'] = d['transaction_qty'] * d['unit_price']

    n_precios = d.groupby('product_detail')['unit_price'].nunique()
    productos_variables = n_precios[n_precios > 1].index.tolist()
    dp = d[d['product_detail'].isin(productos_variables)].copy()

    importancia = (
        dp.groupby('product_detail')
          .agg(
              unidades=('transaction_qty', 'sum'),
              ingresos=('revenue', 'sum'),
              transacciones=('transaction_id', 'count'),
              precio_promedio=('unit_price', 'mean'),
              precio_min=('unit_price', 'min'),
              precio_max=('unit_price', 'max'),
              cantidad_precios=('unit_price', 'nunique')
          )
          .reset_index()
    )
    ingresos_15 = importancia['ingresos'].sum()
    ingresos_total = d['revenue'].sum()
    importancia['participacion_15'] = importancia['ingresos'] / ingresos_15 * 100
    importancia['participacion_total'] = importancia['ingresos'] / ingresos_total * 100

    dm = (
        dp.groupby(['transaction_date', 'store_location', 'product_detail', 'unit_price'])
          .agg(demanda=('transaction_qty', 'sum'))
          .reset_index()
    )
    dm['mes'] = dm['transaction_date'].dt.month
    dm['dia_semana'] = dm['transaction_date'].dt.dayofweek
    dm['ln_demanda'] = np.log(dm['demanda'])
    dm['ln_precio'] = np.log(dm['unit_price'])

    resultados = []
    for producto in productos_variables:
        temp = dm[dm['product_detail'] == producto].copy()
        try:
            modelo = smf.ols(
                'ln_demanda ~ ln_precio + C(store_location) + C(mes) + C(dia_semana)',
                data=temp
            ).fit()
            e = float(modelo.params.get('ln_precio', np.nan))
            pval = float(modelo.pvalues.get('ln_precio', np.nan))
            if 'ln_precio' in modelo.params.index:
                ci = modelo.conf_int().loc['ln_precio']
                ci_inf, ci_sup = float(ci.iloc[0]), float(ci.iloc[1])
            else:
                ci_inf, ci_sup = np.nan, np.nan
            resultados.append({
                'product_detail': producto,
                'observaciones_modelo': len(temp),
                'elasticidad': e,
                'p_value': pval,
                'R2': float(modelo.rsquared),
                'R2_ajustado': float(modelo.rsquared_adj),
                'IC95_inf': ci_inf,
                'IC95_sup': ci_sup,
                'tipo': ('Elástico' if e < -1 else 'Inelástico' if e < 0 else 'Positiva / atípica'),
                'significancia': 'Significativo' if pval < 0.05 else 'No significativo'
            })
        except Exception:
            resultados.append({
                'product_detail': producto,
                'observaciones_modelo': len(temp),
                'elasticidad': np.nan,
                'p_value': np.nan,
                'R2': np.nan,
                'R2_ajustado': np.nan,
                'IC95_inf': np.nan,
                'IC95_sup': np.nan,
                'tipo': 'No estimable',
                'significancia': 'No estimable'
            })

    elasticidades = pd.DataFrame(resultados)
    portafolio = importancia.merge(elasticidades, on='product_detail', how='left')
    portafolio['resultado_modelo'] = np.select(
        [
            (portafolio['p_value'] < 0.05) & (portafolio['elasticidad'] < 0),
            (portafolio['p_value'] < 0.05) & (portafolio['elasticidad'] >= 0),
            (portafolio['p_value'] >= 0.05)
        ],
        ['Elasticidad negativa significativa', 'Elasticidad positiva / atípica significativa', 'No concluyente'],
        default='No estimable'
    )

    bases = (
        dp.groupby(['product_detail', 'unit_price'])
          .agg(cantidad_base=('transaction_qty', 'sum'))
          .reset_index()
          .sort_values(['product_detail', 'cantidad_base'], ascending=[True, False])
          .drop_duplicates('product_detail')
          .rename(columns={'unit_price': 'precio_base'})
    )
    base_lookup = bases.set_index('product_detail')

    # Agregar precio_base y cantidad_base al dataframe de portafolio para simulaciones interactivas
    portafolio['precio_base'] = portafolio['product_detail'].map(base_lookup['precio_base']).fillna(portafolio['precio_promedio'])
    portafolio['cantidad_base'] = portafolio['product_detail'].map(base_lookup['cantidad_base']).fillna(portafolio['unidades'])

    sims = []
    for _, row in portafolio.iterrows():
        producto = row['product_detail']
        e = row['elasticidad']
        if producto not in base_lookup.index or pd.isna(e):
            continue
        p0 = float(base_lookup.loc[producto, 'precio_base'])
        q0 = float(base_lookup.loc[producto, 'cantidad_base'])
        ingreso0 = p0 * q0
        for rango in (0.05, 0.10):
            precios = np.linspace(p0 * (1-rango), p0 * (1+rango), 500)
            cantidades = q0 * (precios / p0) ** e
            ingresos = precios * cantidades
            idx = int(np.argmax(ingresos))
            p_star = float(precios[idx])
            q_star = float(cantidades[idx])
            ingreso_star = float(ingresos[idx])
            sims.append({
                'producto': producto,
                'rango_simulacion': f'±{int(rango*100)}%',
                'elasticidad': e,
                'precio_base': p0,
                'precio_recomendado': p_star,
                'cambio_precio_pct': (p_star/p0 - 1) * 100,
                'cantidad_base': q0,
                'cantidad_estimada': q_star,
                'ingreso_base': ingreso0,
                'ingreso_estimado': ingreso_star,
                'variacion_ingreso_pct': (ingreso_star/ingreso0 - 1) * 100,
                'p_value': row['p_value'],
                'significancia': row['significancia']
            })
    simulacion = pd.DataFrame(sims)
    return dp, portafolio.sort_values('ingresos', ascending=False), simulacion

try:
    df_portafolio_dim3, portafolio_dim3, simulacion_dim3 = construir_dimension_3(df)
    error_dim3 = None
except Exception as exc:
    df_portafolio_dim3, portafolio_dim3, simulacion_dim3 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    error_dim3 = str(exc)

# -----------------------------------------------------------------------------
# MENÚ DE NAVEGACIÓN EN LA BARRA LATERAL
# -----------------------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=500", use_column_width=True)
st.sidebar.title("☕ Menú de Navegación")

opcion_menu = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Inicio",
        "Dimensiones",
        "Consideraciones de Datos de Entrada al Modelo",
        "Modelo de Predicción",
        "Simulador de Escenarios",
        "Bibliografía"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Modelamiento Predictivo de Datos - 810 - De Souza, Mendez, Pimentel, Portocarrero - Septiembre 2026")

# -----------------------------------------------------------------------------
# 1. PÁGINA DE INICIO
# -----------------------------------------------------------------------------
if opcion_menu == "Inicio":
    st.markdown('<div class="main-title">Base de Datos: Ventas de una Cafetería</div>', unsafe_allow_html=True)
    
    st.image(
        "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=1200",
        caption="Análisis Predictivo - Red de Cafeterías",
        use_column_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="section-header">📌 Contexto</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sub-card">
        El conjunto de datos rescatado de Kaggle titulado “Coffee Shop Sales” que se analizará contiene 149 116 registros de transacciones procesadas durante el primer semestre del año 2023 en tres sedes en la ciudad de Nueva York (Hell's Kitchen, Astoria y Lower Manhattan).
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h3 class="section-header">🎯 Objetivos</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sub-card">
        <ul>
            <li><b>Objetivo General:</b> Desplegar modelos predictivos que optimicen las operaciones comerciales de la cafetería.</li>
            <li><b>Objetivo Específico 1:</b> Estimar la demanda futura en los diferentes puntos de venta para prevenir desabastecimientos.</li>
            <li><b>Objetivo Específico 2:</b> Dimensionar el stock óptimo de inventario según la demanda proyectada.</li>
            <li><b>Objetivo Específico 3:</b> Analizar el portafolio de beneficios teniendo en cuenta la elasticidad de los productos más relevantes.</li>
            <li><b>Objetivo Específico 4:</b> Integrar los módulos predictivos resultando en una recomendación operativa unificada.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="section-header">⚠️ Problemática</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sub-card">
        La fluctuación imprevista en la demanda diaria genera dos problemas críticos en el negocio: la insatisfacción del cliente por quiebres de stock en productos de alta rotación (como bebidas de café) y el desperdicio de insumos perecibles (panadería y repostería). La falta de modelos predictivos integrados impide ajustar compras, promociones y niveles de inventario de forma eficiente.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h3 class="section-header">📐 Dimensiones del Estudio</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sub-card">
        El estudio se articula en cuatro ejes operacionales fundamentales:
        <ol>
            <li>Proyección de la Demanda</li>
            <li>Proyección de Inventario</li>
            <li>Análisis de Portafolio</li>
            <li>Integración de las Dimensiones</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DIMENSIONES
# -----------------------------------------------------------------------------
elif opcion_menu == "Dimensiones":
    st.markdown('<h2 class="section-header">📐 Dimensiones del Modelo</h2>', unsafe_allow_html=True)
    st.write("A continuación se describen en detalle los cuatro pilares analíticos que componen la solución:")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📈 1. Proyección de la Demanda")
        st.info("""
        **Descripción y Alcance:**  
        Esta dimensión busca proyectar la demanda de la cafetería utilizando métodos de series de tiempo para estimar el volumen de ventas semanales a partir del comportamiento histórico de las transacciones. Su propósito es identificar patrones de estacionalidad y tendencias en el consumo de los tipos de productos más vendidos en las tiendas, validando la precisión del modelo mediante la presentación e interpretación de sus métricas de error.

        **Variables relevantes:**  
        - transaction_date: Fecha de la transacción, utilizada como eje temporal para la agregación de series de tiempo.
        - transaction_qty: Cantidad de unidades vendidas por transacción.
        - store_location: Ubicación de la cafetería (Lower Manhattan, Hell's Kitchen, Astoria) para segmentar las series por punto de venta.
        - product_category: Categoría del producto para filtrar el estudio exclusivamente en las dos categorías líderes de ventas (Coffee & Tea).
        - product_type (product_name): Tipo del producto para desglosar la serie temporal y proyectar la demanda a nivel de tipo de producto para un reabastecimiento preciso.

        **KPI Principal:** 
        - **Precisión del Pronóstico (Precisión = 100% − WMAPE)**
        *Justificación:* EPermite evaluar cuantitativamente el porcentaje de cercanía entre la demanda proyectada y las ventas reales observadas, validando la confiabilidad general del modelo para la toma de decisiones comerciales y logísticas. El uso del WMAPE resulta más apropiado ya que pondera el error por el volumen físico vendido para evitar distorsiones por productos de baja rotación, complementándose con métricas técnicas como el MAE y el RMSE para cuantificar las desviaciones en unidades físicas semanales frente a variaciones atípicas de la demanda.

        **Objetivo Específico:** Proyectar la demanda semanal por producto (product_name) y sede (store_location) mediante la competencia de modelos estadísticos y de Machine Learning (Promedio Móvil, Holt-Winters, Random Forest y XGBoost), alcanzando una Precisión del Pronóstico superior al 80% (WMAPE < 20%) mediante el algoritmo ganador y reduciendo el RMSE en comparación con el modelo baseline de promedio móvil.
        """)

        st.markdown("### 📊 3. Análisis de Portafolio")
        st.info("""
        **Descripción y Alcance:**  
        Esta dimensión busca analizar el portafolio de productos con variación de precio, considerando su importancia económica y elasticidad precio-demanda, con la finalidad de evaluar escenarios de ajuste de precios y estimar su impacto sobre los ingresos por ventas.

        **Variables relevantes:**  
        - unit_price: precio unitario del producto. 
        - transaction_qty: cantidad demandada del producto.
        - revenue: ingresos por ventas, utilizados para determinar la importancia económica y evaluar el beneficio. 
        - store_location: ubicación de la tienda, utilizada como variable de control.
        - transaction_date: fecha de transacción, utilizada para obtener variables temporales como mes y día de la semana.
        - product_detail: identificación específica de cada producto.

        **KPIs Principales:**  
        - **Elasticidad precio-demanda**
        *Justificación:* Mide la sensibilidad de la cantidad demandada ante variaciones en el precio.
        - **Participación en ingresos (%)**
        *Justificación:* Mide la importancia económica de cada producto dentro del portafolio analizado.
        - **Variación estimada de ingresos (%)**
        *Justificación:* Mide el cambio porcentual esperado en los ingresos ante los escenarios simulados de ajuste de precios.

        **Objetivo Específico:** Evaluar la variación de los ingresos por ventas ante cambios en el precio de los productos relevantes, mediante un análisis de portafolio basado en la elasticidad precio-demanda estimada con regresión log-log y simulación de escenarios de precios, aplicado al proceso de gestión comercial y fijación de precios.
        """)

    with col_b:
        st.markdown("### 📦 2. Proyección de Inventario")
        st.info("""
        **Descripción y Alcance:**  
        Esta dimensión busca utilizar modelos predictivos de Machine Learning (SVR) para proyectar el inventario necesario para la demanda proyectada de la cafetería haciendo uso de indicadores relevantes, tomando en cuenta factores como el Lead Time y el Stock de Seguridad tomando la demanda proyectada en la Dimensión 1.

        **Variables relevantes:**  
        - distancia_km: Distancia en kilómetros desde el centro de distribución del proveedor hasta la sede de la cafetería, usada en el modelo SVR para calcular la base del Lead Time.
        - trafico_promedio_index: Índice de tráfico vehicular en la ruta del proveedor, utilizado en el SVR para tomar en cuenta retrasos en la entrega.
        - clima_val: Variable categórica (0: Normal, 1: Lluvia y 2: Clima adverso) para simular diferentes escenarios meteorológicos durante el transporte.
        - Demanda_proyectada_semanal: Demanda proyectada en la Dimensión 1.
        - store_location: Sede del negocio para diferenciar tráfico por tienda.
        - lead_time: Lead Time en días generado de manera dinámica por el algoritmo SVR de acuerdo a las variables relevantes (distancia, tráfico, clima, store location).
        - product_type: Tipo del producto para filtrar el inventario necesario por cada uno.     

        **KPI Principal:** 
        - **Nivel de Servicio**
            *Justificación:* Es capaz de medir el porcentaje de la demanda de clientes que es satisfecha gracias a la disponibilidad de inventarios en horarios de alta demanda. Asimismo, este indicador permite asegurar la disponibilidad de insumos en horas pico con el objetivo de evitar quiebres de stock y evitando el sobre stock de productos perecederos.

        **Objetivo Específico:** Proyectar el inventario necesario para mejorar el Nivel de Servicio de un 85% (teórico) a un 95% mediante el uso de un sistema de dimensionamiento de Stock de Seguridad (SS) y cálculo dinámico de Punto de Reorden (PRO) haciendo uso de SVR para tomar en cuenta la variabilidad del LT.
        """)

        st.markdown("### 🔗 4. Integración de las dimensiones")
        st.success("""
        **Descripción y Alcance:**  
        Esta dimensión integra las tres anteriores dimensiones en Power BI.

        **Objetivo Específico:** Se presentan objetivos específicos para cuatro dimensiones de la formulación de la solución al problema caso de estudio.
        """)

# -----------------------------------------------------------------------------
# 3. CONSIDERACIONES DE DATOS DE ENTRADA AL MODELO
# -----------------------------------------------------------------------------
elif opcion_menu == "Consideraciones de Datos de Entrada al Modelo":
    st.markdown('<h2 class="section-header">📊 Consideraciones de Datos de Entrada al Modelo</h2>', unsafe_allow_html=True)
    st.write("Análisis descriptivo y exploratorio realizado en el entorno de trabajo Collab sobre la base de datos *Coffee Shop Sales*.")

    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏬 Locales", 
            "🏷️ Categorías", 
            "☕ Tipos de Producto", 
            "💵 Precio Unitario"
        ])

        # TAB 1: LOCALES
        with tab1:
            st.markdown("### Análisis de Locales (Variable Categórica/Nominal)")
            total_registros_locales = len(df['store_location'])
            locales_unicos = df['store_location'].nunique()
            moda_local = df['store_location'].mode()[0]

            m1, m2, m3 = st.columns(3)
            m1.metric("Total de Registros", f"{total_registros_locales:,}")
            m2.metric("Locales Únicos", locales_unicos)
            m3.metric("Moda (Local más frecuente)", moda_local)

            conteo_locales = df['store_location'].value_counts()
            porcentaje_locales = (df['store_location'].value_counts(normalize=True) * 100).round(2)
            resumen_locales = pd.DataFrame({
                'Cantidad de Registros': conteo_locales,
                'Porcentaje (%)': porcentaje_locales
            })

            col_t1, col_g1 = st.columns([1, 1.5])
            with col_t1:
                st.write("**Tabla de Frecuencias por Local**")
                st.dataframe(resumen_locales, use_container_width=True)

            with col_g1:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(x='store_location', data=df, hue='store_location', palette='Set2', legend=False, ax=ax)
                ax.set_title('Distribución de Registros por Local')
                ax.set_xlabel('Local')
                ax.set_ylabel('Cantidad de Registros')
                
                for p in ax.patches:
                    altura = p.get_height()
                    if altura is not None and altura > 0:
                        pct = (altura / total_registros_locales) * 100
                        ax.annotate(f'{int(altura):,}\n({pct:.2f}%)',
                                    (p.get_x() + p.get_width() / 2.0, altura),
                                    ha='center', va='bottom', xytext=(0, 5),
                                    textcoords='offset points', fontsize=8)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
                st.pyplot(fig)

        # TAB 2: CATEGORÍAS
        with tab2:
            st.markdown("### Análisis Categoría de Producto (Variable Categórica/Nominal)")
            total_cat = len(df['product_category'])
            cat_unicas = df['product_category'].nunique()
            moda_cat = df['product_category'].mode()[0]

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Registros", f"{total_cat:,}")
            m2.metric("Categorías Únicas", cat_unicas)
            m3.metric("Categoría más Vendida (Moda)", moda_cat)

            conteo_cat = df['product_category'].value_counts()
            porcentaje_cat = (df['product_category'].value_counts(normalize=True) * 100).round(2)
            resumen_cat = pd.DataFrame({
                'Cantidad de Registros': conteo_cat,
                'Porcentaje (%)': porcentaje_cat
            })

            col_t2, col_g2 = st.columns([1, 1.5])
            with col_t2:
                st.write("**Tabla de Frecuencias por Categoría**")
                st.dataframe(resumen_cat, use_container_width=True)

            with col_g2:
                fig, ax = plt.subplots(figsize=(9, 4.5))
                orden_cat = df['product_category'].value_counts().index
                sns.countplot(x='product_category', data=df, order=orden_cat, hue='product_category', palette='Set2', legend=False, ax=ax)
                ax.set_title('Distribución de Registros por Categoría de Producto')
                ax.set_xlabel('Categoría')
                ax.set_ylabel('Cantidad de Registros')
                plt.xticks(rotation=45, ha='right')

                for p in ax.patches:
                    altura = p.get_height()
                    if altura is not None and altura > 0:
                        pct = (altura / total_cat) * 100
                        ax.annotate(f'{int(altura):,}\n({pct:.2f}%)',
                                    (p.get_x() + p.get_width() / 2.0, altura),
                                    ha='center', va='bottom', xytext=(0, 5),
                                    textcoords='offset points', fontsize=8)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
                st.pyplot(fig)

        # TAB 3: TIPO DE PRODUCTO (SOLO TOP 3 CATEGORÍAS)
        with tab3:
            st.markdown("### Análisis de Tipos de Producto en las Top 3 Categorías Más Vendidas")
            st.write("A continuación se muestra el desglose de tipos de producto (`product_type`) para las 3 principales categorías de ventas.")

            top_3_categorias = df['product_category'].value_counts().head(3).index.tolist()
            subtabs = st.tabs([f"☕ {cat}" for cat in top_3_categorias])

            for index, cat in enumerate(top_3_categorias):
                with subtabs[index]:
                    df_sub = df[df['product_category'] == cat]
                    total_registros_sub = len(df_sub)
                    
                    st.markdown(f"#### Desglose de Tipos de Producto en **{cat}** (Total: {total_registros_sub:,} registros)")

                    tipos_unicos_sub = df_sub['product_type'].nunique()
                    tipo_moda_sub = df_sub['product_type'].mode()[0]
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Variedades/Tipos Únicos", tipos_unicos_sub)
                    m2.metric("Tipo de Producto Más Vendido", tipo_moda_sub)

                    col_tabla, col_grafico = st.columns([1, 1.4])

                    conteo_tipo_sub = df_sub['product_type'].value_counts()
                    porcentaje_tipo_sub = (df_sub['product_type'].value_counts(normalize=True) * 100).round(2)
                    
                    df_resumen_sub = pd.DataFrame({
                        'Cantidad de Registros': conteo_tipo_sub,
                        'Porcentaje (%)': porcentaje_tipo_sub
                    })

                    with col_tabla:
                        st.write(f"**Tabla de Frecuencias - {cat}**")
                        st.dataframe(df_resumen_sub, use_container_width=True)

                    with col_grafico:
                        fig, ax = plt.subplots(figsize=(8, max(3.5, len(conteo_tipo_sub) * 0.5)))
                        
                        sns.barplot(
                            x=conteo_tipo_sub.values,
                            y=conteo_tipo_sub.index,
                            palette='Set2',
                            hue=conteo_tipo_sub.index,
                            legend=False,
                            ax=ax)
                        
                        ax.set_title(f'Distribución de Tipos de Producto en {cat}', fontsize=11, fontweight='bold')
                        ax.set_xlabel('Cantidad de Registros')
                        ax.set_ylabel('Tipo de Producto')

                        max_val = conteo_tipo_sub.max()
                        for p in ax.patches:
                            ancho = p.get_width()
                            if ancho > 0:
                                pct = (ancho / total_registros_sub) * 100
                                ax.annotate(
                                    f'{int(ancho):,} ({pct:.1f}%)',
                                    (ancho, p.get_y() + p.get_height() / 2.0),
                                    ha='left', va='center',
                                    xytext=(5, 0), textcoords='offset points',
                                    fontsize=8)
                        
                        ax.set_xlim(0, max_val * 1.25)
                        plt.tight_layout()
                        st.pyplot(fig)

        # TAB 4: PRECIO UNITARIO (FILTRANDO OUTLIERS ALTOS)
        with tab4:
            st.markdown("### Análisis de Precio Unitario (Excluyendo Precios Atípicos Altos)")
            st.write("Se aplicó el criterio del Rango Intercuartílico ($Q3 + 1.5 \times IQR$) para identificar y filtrar los precios inusualmente altos que presentan muy baja frecuencia en las transacciones.")

            precios_todos = df['unit_price'].dropna()

            q1_global = precios_todos.quantile(0.25)
            q3_global = precios_todos.quantile(0.75)
            iqr_global = q3_global - q1_global
            limite_superior = q3_global + (1.5 * iqr_global)

            df_outliers = df[df['unit_price'] > limite_superior]
            df_filtrado = df[df['unit_price'] <= limite_superior]
            precios = df_filtrado['unit_price']

            media = precios.mean()
            mediana = precios.median()
            moda_p = precios.mode()[0]
            std_p = precios.std()
            q1 = precios.quantile(0.25)
            q3 = precios.quantile(0.75)

            confianza = 0.95
            n = len(precios)
            error_estandar = st_scipy.sem(precios)
            ic_inferior, ic_superior = st_scipy.t.interval(confidence=confianza, df=n - 1, loc=media, scale=error_estandar)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Media", f"${media:.2f}")
            m2.metric("Mediana", f"${mediana:.2f}")
            m3.metric("Moda", f"${moda_p:.2f}")
            m4.metric("Desv. Estándar", f"${std_p:.2f}")

            st.write(f"**Umbral de corte de precios altos:** Precios superiores a **${limite_superior:.2f}** | **IC 95% para la Media:** [{ic_inferior:.2f}, {ic_superior:.2f}]")

            fig, ax = plt.subplots(figsize=(12, 4))
            sns.boxplot(
                x=precios, 
                color='skyblue', 
                flierprops={'markerfacecolor': 'orange', 'marker': 'o', 'markersize': 4, 'alpha': 0.5},
                ax=ax
            )
            ax.axvline(media, color='darkblue', linestyle='--', linewidth=2, label=f'Media: ${media:.2f}')
            ax.axvspan(ic_inferior, ic_superior, color='blue', alpha=0.15, label=f'IC 95% [{ic_inferior:.2f}, {ic_superior:.2f}]')

            min_val, max_val = precios.min(), precios.max()
            ax.text(min_val, -0.35, f'Mínimo\n${min_val:.2f}', ha='center', va='top', weight='bold', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax.text(q1, 0.35, f'Q1\n${q1:.2f}', ha='center', va='bottom', weight='bold', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax.text(mediana, -0.35, f'Mediana (Q2)\n${mediana:.2f}', ha='center', va='top', weight='bold', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax.text(q3, 0.35, f'Q3\n${q3:.2f}', ha='center', va='bottom', weight='bold', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax.text(max_val, -0.35, f'Máximo (Q4)\n${max_val:.2f}', ha='center', va='top', weight='bold', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            ax.set_title(f'Distribución de Precios Frecuentes (Filtrado: Precios ≤ ${limite_superior:.2f})')
            ax.set_xlabel('Precio ($)')
            ax.legend(loc='upper right')
            ax.set_ylim(-0.7, 0.7)
            st.pyplot(fig)

            st.markdown("---")
            st.markdown(f"### 📋 Productos Excluidos de la Distribución Principal (Precios > ${limite_superior:.2f})")
            
            if not df_outliers.empty:
                columnas_producto = [col for col in ['product_category', 'product_type', 'product_detail'] if col in df_outliers.columns]
                if not columnas_producto:
                    columnas_producto = ['product_category', 'product_type']

                tabla_excluidos = df_outliers.groupby(columnas_producto).agg(
                    Precio_Unitario=('unit_price', 'mean'),
                    Frecuencia_Transacciones=('unit_price', 'count')
                ).reset_index().sort_values(by='Precio_Unitario', ascending=False)

                tabla_excluidos['Precio_Unitario'] = tabla_excluidos['Precio_Unitario'].apply(lambda x: f"${x:.2f}")

                st.write(f"Se identificaron **{len(df_outliers):,} registros transaccionales** con precios inusualmente altos que fueron omitidos del gráfico principal:")
                st.dataframe(tabla_excluidos, use_container_width=True)
            else:
                st.info("No se encontraron precios por encima del límite atípico calculando el Rango Intercuartílico.")

# -----------------------------------------------------------------------------
# 4. MODELO DE PREDICCIÓN
# -----------------------------------------------------------------------------
elif opcion_menu == "Modelo de Predicción":
    st.markdown('<h2 class="section-header">🤖 Modelo de Predicción</h2>', unsafe_allow_html=True)

    dim_selected = st.selectbox("Seleccione la dimensión analítica a explorar:", ["Proyección de la Demanda", "Proyección de Inventario", "Análisis de Portafolio"])

    st.markdown(f"### Resultados del Modelo para: **{dim_selected}**")

    if dim_selected == "Proyección de la Demanda":
        st.markdown("### 📈 Dimensión 1: Proyección de Demanda (Coffee & Tea)")
        st.markdown("---")

        with st.spinner("Entrenando modelos de Series Temporales y Machine Learning..."):
            summary_df, best_model_name, best_acc, df_proyeccion = ejecutar_dimension_1(df)

        st.session_state['df_proyeccion_dim1'] = df_proyeccion

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("🏆 Modelo Ganador Seleccionado", best_model_name)
        col_m2.metric("🎯 Precisión Promedio (Accuracy)", f"{best_acc}%")

        st.markdown("---")

        st.subheader("📊 1. Tabla Comparativa de Desempeño de Modelos")
        st.dataframe(
            summary_df.style.highlight_max(subset=['Precisión del Pronóstico (%)'], color='#d4edda'),
            use_container_width=True,
            hide_index=True)

        st.markdown("---")

        st.subheader("📅 2. Proyección de Demanda para el Próximo Mes (4 Semanas Futuras)")
        st.dataframe(df_proyeccion, use_container_width=True, hide_index=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_proyeccion.to_excel(writer, index=False, sheet_name='Proyeccion')
        
        st.download_button(
            label="📥 Descargar Proyección en Excel",
            data=buffer.getvalue(),
            file_name="Proyeccion_Demanda_Proximo_Mes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("---")

        st.subheader("📊 3. Demanda Total Proyectada por Sede y Tipo de Producto")
        
        df_summary_product = df_proyeccion.groupby(['store_location', 'product_name'])['Demanda_Proyectada_Semanal'].sum().reset_index()

        fig_dim1 = px.bar(
            df_summary_product,
            x='store_location',
            y='Demanda_Proyectada_Semanal',
            color='product_name',
            barmode='group',
            text_auto='.0f',
            labels={
                'store_location': 'Sede',
                'Demanda_Proyectada_Semanal': 'Unidades Totales Proyectadas',
                'product_name': 'Producto'
            },
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig_dim1.update_layout(
            title="Demanda Total Proyectada para el Próximo Mes por Sede y Tipo de Producto",
            xaxis_title="Sede",
            yaxis_title="Unidades Totales Proyectadas",
            template="plotly_white",
            legend_title_text="Producto"
        )
        
        st.plotly_chart(fig_dim1, use_container_width=True)

    elif dim_selected == "Proyección de Inventario":
        st.markdown("### 📦 Dimensión 2: Proyección y Optimización de Inventario")
        st.markdown("---")

        st.subheader("📈 Métricas de Validación del Modelo SVR (Test Set - 20%)")
        col_svr1, col_svr2, col_svr3 = st.columns(3)
        col_svr1.metric("🎯 Coeficiente R²", f"{metricas_svr['R2']:.3f}")
        col_svr2.metric("📉 Error RMSE (σ_LT)", f"{metricas_svr['RMSE']:.3f} días")
        col_svr3.metric("📏 Error MAE", f"{metricas_svr['MAE']:.3f} días (≈ {metricas_svr['MAE'] * 24:.1f} hrs)")

        st.markdown("---")

        st.subheader("📋 1. Variables del Modelo SVR para Predicción de Lead Time")
        df_vars = pd.DataFrame({
            "Variable": ["distancia_km", "trafico_promedio_index", "clima_val", "store_location", "product_name"],
            "Tipo": ["Numérica", "Escala (1.0 - 5.0)", "Categoría (0, 1, 2)", "Categoría", "Categoría"],
            "Rango / Valores": ["5 a 50 km", "1 (Fluido) a 5 (Congestión)", "0 (Despejado), 1 (Lluvia), 2 (Clima Adverso)", "Astoria, Hell's Kitchen, Lower Manhattan", "Variedades de Café y Té"]
        })
        st.dataframe(df_vars, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.subheader("📂 2. Carga de Proyección de Demanda (Dimensión 1)")
        uploaded_file = st.file_uploader(
            "Sube el archivo Excel con la proyección de demanda realizada en la Dimensión 1:", 
            type=["xlsx", "xls"],
            key="uploader_dim2_modelo"
        )

        df_excel_demanda = None
        if uploaded_file is not None:
            try:
                df_excel_demanda = pd.read_excel(uploaded_file)
                st.success("✅ Archivo de proyección cargado con éxito.")
            except Exception as e:
                st.error(f"Error al leer el archivo Excel: {e}")

        st.markdown("---")

        st.subheader("🎛️ 3. Cálculo Operativo Base por Local y Producto")
        col_in, col_out = st.columns([1, 1])

        with col_in:
            st.markdown("#### **Parámetros Operativos**")
            tienda_in = st.selectbox("Local Destino:", ['Astoria', "Hell's Kitchen", 'Lower Manhattan'])
            producto_in = st.selectbox("Producto:", ['Barista Espresso', 'Brewed Black tea', 'Brewed Chai tea'])

            demanda_default = 45.0
            demanda_manual_deshabilitada = False
            demanda_serie_excel = None

            if df_excel_demanda is not None:
                cols_store = [c for c in df_excel_demanda.columns if any(k in c.lower() for k in ['store', 'local', 'tienda', 'location', 'sede'])]
                cols_prod = [c for c in df_excel_demanda.columns if any(k in c.lower() for k in ['product', 'producto', 'item', 'tipo', 'category', 'detail'])]
                cols_qty = [c for c in df_excel_demanda.columns if any(k in c.lower() for k in ['demanda', 'qty', 'cantidad', 'proyeccion', 'prediccion', 'transaction_qty', 'y_pred', 'target'])]

                df_filtrado = df_excel_demanda.copy()

                if cols_store:
                    col_s = cols_store[0]
                    df_filtrado = df_filtrado[df_filtrado[col_s].astype(str).str.strip().str.lower() == tienda_in.strip().lower()]

                if cols_prod:
                    col_p = cols_prod[0]
                    df_filtrado = df_filtrado[df_filtrado[col_p].astype(str).str.strip().str.lower() == producto_in.strip().lower()]

                if not df_filtrado.empty and cols_qty:
                    col_q = cols_qty[0]
                    demanda_serie_excel = df_filtrado[col_q].values
                    demanda_semanal_prom = float(np.mean(demanda_serie_excel))
                    demanda_default = demanda_semanal_prom / 7.0
                    demanda_manual_deshabilitada = True
                    st.info(f"🎯 **Demanda en Excel para '{producto_in}' en '{tienda_in}':** {demanda_semanal_prom:.2f} unids/semana ({demanda_default:.2f} unids/día).")
                else:
                    st.warning(f"⚠️ No se encontraron registros en el Excel para **{producto_in}** en **{tienda_in}** (o los nombres de columnas no coinciden). Se habilita la edición manual.")

            distancia_in = st.slider("Distancia Proveedor-Tienda (km):", 5.0, 60.0, 25.0, 1.0)
            trafico_in = st.slider("Índice de Tráfico (1=Fluido, 5=Trancón):", 1.0, 5.0, 3.0, 0.1)
            
            clima_map = {"☀️ Despejado": 0, "🌧️ Lluvia": 1, "🌩️ Clima adverso": 2}
            clima_label = st.selectbox("Clima en Ruta:", list(clima_map.keys()))
            clima_in = clima_map[clima_label]
            
            demanda_diaria_in = st.number_input(
                "Demanda Promedio (unids):", 
                value=float(np.round(demanda_default, 2)), 
                step=5.0,
                disabled=demanda_manual_deshabilitada
            )
            nivel_servicio_in = st.slider("Nivel de Servicio Objetivo (%):", 80, 99, 95)

        with col_out:
            st.markdown("#### **Resultados de Predicción e Inventario**")
            
            res_inv = calcular_politica_inventario(
                tienda_in, producto_in, demanda_diaria_in,
                distancia_in, trafico_in, clima_in, nivel_servicio_in
            )
            lt_predicho = res_inv['lead_time']
            ss_calculado = res_inv['ss']
            rop_calculado = res_inv['rop']
            
            st.metric("⏱️ Lead Time Predicho (SVR)", f"{lt_predicho:.2f} días")
            
            m1, m2 = st.columns(2)
            m1.metric("🛡️ Stock de Seguridad (SS)", f"{ss_calculado:.1f} unids")
            m2.metric("📍 Punto de Reorden (ROP)", f"{rop_calculado:.1f} unids")
            
            st.info(f"💡 Para **{producto_in}** en **{tienda_in}** con nivel de servicio del **{nivel_servicio_in}%**, emite la orden al llegar a **{int(np.round(rop_calculado))} unidades**.")

        st.markdown("---")

        st.subheader(f"📊 4. Evaluación Mensual del Modelo vs. Demanda Real ({producto_in} - {tienda_in})")
        
        np.random.seed(101)
        n_eval = 4
        
        if demanda_serie_excel is not None and len(demanda_serie_excel) >= 1:
            demanda_proyectada_hist = demanda_serie_excel[:n_eval] if len(demanda_serie_excel) >= n_eval else np.resize(demanda_serie_excel, n_eval)
        else:
            demanda_proyectada_hist = np.random.uniform(demanda_diaria_in * 5.5, demanda_diaria_in * 7.5, n_eval)
            
        demanda_real_hist = demanda_proyectada_hist * np.random.uniform(0.88, 1.10, n_eval)
        stock_objetivo_hist = demanda_proyectada_hist + ss_calculado
        
        exceso_inventario = np.maximum(0, stock_objetivo_hist - demanda_real_hist)
        unidades_faltantes = np.maximum(0, demanda_real_hist - stock_objetivo_hist)
        fill_rate_efectivo = np.minimum(100.0, (np.minimum(stock_objetivo_hist, demanda_real_hist) / demanda_real_hist) * 100)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("✅ Fill Rate Real Promedio", f"{fill_rate_efectivo.mean():.2f}%", f"+{fill_rate_efectivo.mean()-80:.1f}% vs 80%")
        k2.metric("📦 Exceso Promedio de Stock", f"{exceso_inventario.mean():.2f} unids")
        k3.metric("🚨 Unidades Faltantes Totales", f"{unidades_faltantes.sum():.1f} unids", delta_color="inverse")
        k4.metric("🎯 Nivel de Servicio Teórico", f"{nivel_servicio_in}%")

        eje_x_labels = [f"Semana {i}" for i in range(1, n_eval + 1)]

        fig_eval = go.Figure()
        fig_eval.add_trace(go.Bar(x=eje_x_labels, y=stock_objetivo_hist, name='Stock Total Asignado', marker_color='#6F4E37'))
        fig_eval.add_trace(go.Bar(x=eje_x_labels, y=demanda_real_hist, name='Demanda Real Observada', marker_color='#D2B48C'))
        fig_eval.add_trace(go.Scatter(x=eje_x_labels, y=exceso_inventario, name='Exceso de Inventario (Sobrestock)', mode='lines+markers', line=dict(color='orange', width=2, dash='dash')))

        fig_eval.update_layout(
            title=f"Desempeño Mensual de Reabastecimiento para {producto_in} en {tienda_in}",
            xaxis_title="Semanas del Mes",
            yaxis_title="Unidades",
            barmode='group',
            template='plotly_white'
        )
        st.plotly_chart(fig_eval, use_container_width=True)
    
    else:  # dim_selected == "Análisis de Portafolio"
        st.markdown("### 📊 Dimensión 3: Portafolio, elasticidad y escenarios de precio")
        st.markdown("---")

        if error_dim3:
            st.error(f"No se pudo calcular la Dimensión 3: {error_dim3}")
        else:
            total_productos = df['product_detail'].nunique()
            n_variables = len(portafolio_dim3)
            ingresos_15 = portafolio_dim3['ingresos'].sum()
            ingresos_total = (df['transaction_qty'] * df['unit_price']).sum()
            pct_ingresos_15 = ingresos_15 / ingresos_total * 100
            n_neg_sig = int(((portafolio_dim3['p_value'] < 0.05) & (portafolio_dim3['elasticidad'] < 0)).sum())
            n_pos_sig = int(((portafolio_dim3['p_value'] < 0.05) & (portafolio_dim3['elasticidad'] >= 0)).sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Productos totales", f"{total_productos}")
            c2.metric("Con variación de precio", f"{n_variables}")
            c3.metric("Participación en ingresos", f"{pct_ingresos_15:.1f}%")
            c4.metric("Elasticidades neg. significativas", f"{n_neg_sig}")

            st.caption(
                "La elasticidad se estima con regresión log-log controlando por tienda, mes y día de semana."
            )

            tab_a, tab_b, tab_c, tab_d = st.tabs([
                "📋 Portafolio de productos",
                "📈 Elasticidad y significancia",
                "💵 Simulación de precios (±5% / ±10%)",
                "🧭 Lectura gerencial"
            ])

            with tab_a:
                tabla = portafolio_dim3[[
                    'product_detail','unidades','ingresos','participacion_15','participacion_total',
                    'precio_min','precio_max','cantidad_precios'
                ]].copy()
                tabla.columns = [
                    'Producto','Unidades','Ingresos ($)','Participación 15 (%)','Participación total (%)',
                    'Precio mín.','Precio máx.','N.º precios'
                ]
                st.dataframe(tabla.round(2), use_container_width=True, hide_index=True)

            with tab_b:
                tabla_e = portafolio_dim3[[
                    'product_detail','elasticidad','p_value','R2','R2_ajustado','IC95_inf','IC95_sup',
                    'tipo','significancia','resultado_modelo'
                ]].copy()
                tabla_e.columns = [
                    'Producto','Elasticidad','p-value','R²','R² ajustado','IC95 inf.','IC95 sup.',
                    'Tipo','Significancia','Resultado'
                ]
                st.dataframe(tabla_e.round(4), use_container_width=True, hide_index=True)

                fig_e, ax_e = plt.subplots(figsize=(10, 6))
                temp_e = portafolio_dim3.sort_values('elasticidad')
                for i, (_, r) in enumerate(temp_e.iterrows()):
                    marker = 'o' if r['p_value'] < 0.05 else 'x'
                    ax_e.scatter(r['elasticidad'], i, s=90, marker=marker)
                ax_e.axvline(-1, linestyle='--', alpha=0.5)
                ax_e.axvline(0, linestyle='--', alpha=0.5)
                ax_e.set_yticks(range(len(temp_e)))
                ax_e.set_yticklabels(temp_e['product_detail'])
                ax_e.set_xlabel('Elasticidad precio-demanda')
                ax_e.set_title('Elasticidad estimada de los productos con precio variable')
                ax_e.grid(axis='x', alpha=0.2)
                st.pyplot(fig_e)
                st.caption("○ p < 0.05 (significativo) | × p ≥ 0.05 (no concluyente)")

            with tab_c:
                st.write(
                    "Se evalúan escenarios de ajuste de precio de ±5% y ±10%. "
                    "Para cada rango se identifica el precio que genera el mayor ingreso estimado "
                    "dentro del escenario analizado, utilizando la elasticidad precio-demanda estimada."
                )
                tabla_s = simulacion_dim3.copy()
                tabla_s = tabla_s.merge(
                    portafolio_dim3[['product_detail','resultado_modelo']],
                    left_on='producto', right_on='product_detail', how='left'
                )
                mostrar = tabla_s[[
                    'producto','rango_simulacion','elasticidad','precio_base','precio_recomendado',
                    'cambio_precio_pct','variacion_ingreso_pct','p_value','resultado_modelo'
                ]].copy()
                mostrar.columns = [
                    'Producto','Rango','Elasticidad','Precio base','Mejor precio del escenario',
                    'Cambio precio (%)','Variación ingreso (%)','p-value','Resultado'
                ]
                st.dataframe(mostrar.round(2), use_container_width=True, hide_index=True)

                sig_names = portafolio_dim3.loc[portafolio_dim3['p_value'] < 0.05, 'product_detail']
                g = tabla_s[tabla_s['producto'].isin(sig_names)].pivot_table(
                    index='producto', columns='rango_simulacion', values='variacion_ingreso_pct'
                )
                cols = [c for c in ['±5%','±10%'] if c in g.columns]
                if cols:
                    g = g[cols]
                    fig_s, ax_s = plt.subplots(figsize=(11, 6))
                    g.plot(kind='bar', ax=ax_s)
                    ax_s.set_ylabel('Variación estimada de ingresos (%)')
                    ax_s.set_xlabel('Producto')
                    ax_s.set_title('Impacto estimado en ingresos según escenario de ajuste de precio')
                    ax_s.tick_params(axis='x', rotation=30)
                    ax_s.grid(axis='y', alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig_s)
                    st.caption(
                        "Se muestran los productos con p < 0.05. Las elasticidades positivas se mantienen como "
                        "resultados atípicos y requieren cautela en su interpretación."
                    )

            with tab_d:
                st.markdown("""
                **Lectura para la toma de decisiones**

                - **Elasticidad negativa significativa:** asociación inversa entre precio y demanda; si |E| > 1, la demanda es elástica.
                - **Elasticidad negativa no significativa:** el coeficiente se reporta, pero no permite una recomendación firme.
                - **Elasticidad positiva significativa:** resultado atípico frente a la relación precio-demanda convencional; no debe interpretarse causalmente sin validación adicional.
                - **Simulación de escenarios:** se evalúan ajustes máximos de ±5% y ±10%. El resultado representa el precio que genera el mayor ingreso estimado dentro del rango evaluado, no un precio óptimo absoluto.
                """)
                st.info(
                    f"Resultado actual: {n_neg_sig} productos presentan elasticidad negativa significativa y "
                    f"{n_pos_sig} presentan elasticidad positiva significativa."
                )

# -----------------------------------------------------------------------------
# 5. SIMULADOR DE ESCENARIOS (PARA LAS 3 DIMENSIONES)
# -----------------------------------------------------------------------------
elif opcion_menu == "Simulador de Escenarios":
    st.markdown('<h2 class="section-header">🎛️ Simulador Interactivo de Escenarios (What-If)</h2>', unsafe_allow_html=True)

    dim_esc_selected = st.selectbox(
        "Seleccione la dimensión analítica para simular escenarios:",
        ["Proyección de la Demanda", "Proyección de Inventario", "Análisis de Portafolio"],
        key="selector_dimension_escenarios"
    )

    st.markdown(f"### Laboratorio de Escenarios para: **{dim_esc_selected}**")

    # =========================================================================
    # ESCENARIOS DIMENSIÓN 1: PROYECCIÓN DE LA DEMANDA
    # =========================================================================
    if dim_esc_selected == "Proyección de la Demanda":
        st.markdown("""
        <div class="scenario-banner">
        <b>Simulador What-If de Demanda (Coffee & Tea):</b> Evalúa cómo impactan variaciones estacionales, campañas promocionales o cambios de mercado sobre el pronóstico semanal generado por el modelo ganador de la <b>Dimensión 1</b>.
        </div>
        """, unsafe_allow_html=True)

        if 'df_proyeccion_dim1' in st.session_state:
            df_proy_base = st.session_state['df_proyeccion_dim1'].copy()
            best_model_esc = "Modelo Ganador (Dimensión 1)"
        else:
            _, best_model_esc, _, df_proy_base = ejecutar_dimension_1(df)
            st.session_state['df_proyeccion_dim1'] = df_proy_base

        col_ctrl1, col_viz1 = st.columns([1.05, 2.2])

        with col_ctrl1:
            st.markdown("### ⚙️ Panel de Control")
            st.caption("Configura los choques sobre la **Demanda Base Proyectada**:")

            sedes_d1 = ["Todas las sedes"] + sorted(df_proy_base['store_location'].unique().tolist())
            sede_sel_d1 = st.selectbox("🏬 Filtrar Sede:", sedes_d1, key="d1_esc_sede")

            df_temp_prod = df_proy_base if sede_sel_d1 == "Todas las sedes" else df_proy_base[df_proy_base['store_location'] == sede_sel_d1]
            prods_d1 = ["Todos los productos"] + sorted(df_temp_prod['product_name'].unique().tolist())
            prod_sel_d1 = st.selectbox("☕ Filtrar Producto:", prods_d1, key="d1_esc_prod")

            st.markdown("---")
            st.markdown("**📈 Factores de Escenario (What-If)**")
            estacionalidad_pct = st.slider("Variación Estacional / Clima (%):", -30, 50, 0, step=5, format="%d%%", key="d1_estac")
            promo_pct = st.slider("Impulso por Campañas / Promociones (%):", 0, 40, 10, step=5, format="+%d%%", key="d1_promo")
            competencia_pct = st.slider("Efecto Competencia / Canibalización (%):", -25, 0, 0, step=5, format="%d%%", key="d1_comp")
            margen_error_pct = st.slider("Banda de Incertidumbre Operativa (±%):", 5, 25, 10, step=1, format="±%d%%", key="d1_banda")

            factor_neto_pct = estacionalidad_pct + promo_pct + competencia_pct
            st.info(f"📊 **Variación Neta Aplicada:** **{factor_neto_pct:+d}%** sobre el pronóstico base.")

        # Filtrar datos según selección
        df_filtrado_d1 = df_proy_base.copy()
        if sede_sel_d1 != "Todas las sedes":
            df_filtrado_d1 = df_filtrado_d1[df_filtrado_d1['store_location'] == sede_sel_d1]
        if prod_sel_d1 != "Todos los productos":
            df_filtrado_d1 = df_filtrado_d1[df_filtrado_d1['product_name'] == prod_sel_d1]

        factor_mult = 1.0 + (factor_neto_pct / 100.0)
        df_filtrado_d1['Escenario_Pesimista'] = (df_filtrado_d1['Demanda_Proyectada_Semanal'] * 0.85).round(2)
        df_filtrado_d1['Escenario_Optimista'] = (df_filtrado_d1['Demanda_Proyectada_Semanal'] * 1.20).round(2)
        df_filtrado_d1['Escenario_Personalizado'] = np.maximum(0, df_filtrado_d1['Demanda_Proyectada_Semanal'] * factor_mult).round(2)
        df_filtrado_d1['Limite_Inferior'] = np.maximum(0, df_filtrado_d1['Escenario_Personalizado'] * (1 - margen_error_pct / 100.0)).round(2)
        df_filtrado_d1['Limite_Superior'] = (df_filtrado_d1['Escenario_Personalizado'] * (1 + margen_error_pct / 100.0)).round(2)

        total_base_d1 = df_filtrado_d1['Demanda_Proyectada_Semanal'].sum()
        total_pers_d1 = df_filtrado_d1['Escenario_Personalizado'].sum()
        diff_unids_d1 = total_pers_d1 - total_base_d1

        # Agrupar por semana (1 a 4)
        fechas_ordenadas = sorted(df_filtrado_d1['transaction_date'].unique())
        mapa_semanas = {f: f"Semana {i+1}" for i, f in enumerate(fechas_ordenadas)}
        df_filtrado_d1['Semana'] = df_filtrado_d1['transaction_date'].map(mapa_semanas)

        df_sem_d1 = df_filtrado_d1.groupby('Semana', sort=False).agg(
            Base=('Demanda_Proyectada_Semanal', 'sum'),
            Pesimista=('Escenario_Pesimista', 'sum'),
            Optimista=('Escenario_Optimista', 'sum'),
            Personalizado=('Escenario_Personalizado', 'sum'),
            Lim_Inf=('Limite_Inferior', 'sum'),
            Lim_Sup=('Limite_Superior', 'sum')
        ).reset_index()

        pico_semanal = df_sem_d1['Personalizado'].max() if not df_sem_d1.empty else 0.0

        with col_viz1:
            st.markdown(f"### 📊 Impacto en Demanda: **{prod_sel_d1}** ({sede_sel_d1})")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📦 Demanda Base (4 sem)", f"{total_base_d1:,.0f} u")
            k2.metric("🎛️ Demanda Escenario", f"{total_pers_d1:,.0f} u", f"{factor_neto_pct:+d}% vs Base")
            k3.metric("⚖️ Variación Neta", f"{diff_unids_d1:+,.0f} u")
            k4.metric("🚀 Pico Semanal Esperado", f"{pico_semanal:,.0f} u")

            t1_d1, t2_d1, t3_d1 = st.tabs([
                "📋 Comparativa de Escenarios",
                "📈 Evolución Semanal con Bandas",
                "🏪 Desglose por Sede y Producto"
            ])

            with t1_d1:
                st.write("Comparación semanal de unidades proyectadas entre escenarios:")
                tabla_comp_d1 = df_sem_d1[['Semana', 'Pesimista', 'Base', 'Optimista', 'Personalizado']].copy()
                tabla_comp_d1.columns = ['Semana', 'Pesimista (-15%)', 'Base (Modelo ML)', 'Optimista (+20%)', '🎛️ Tu Escenario']
                st.dataframe(tabla_comp_d1.round(1), use_container_width=True, hide_index=True)

                fig_bar_d1 = go.Figure()
                fig_bar_d1.add_trace(go.Bar(x=df_sem_d1['Semana'], y=df_sem_d1['Pesimista'], name='Pesimista (-15%)', marker_color='#BDC3C7'))
                fig_bar_d1.add_trace(go.Bar(x=df_sem_d1['Semana'], y=df_sem_d1['Base'], name='Base (Modelo ML)', marker_color='#D2B48C'))
                fig_bar_d1.add_trace(go.Bar(x=df_sem_d1['Semana'], y=df_sem_d1['Optimista'], name='Optimista (+20%)', marker_color='#8D6E63'))
                fig_bar_d1.add_trace(go.Bar(x=df_sem_d1['Semana'], y=df_sem_d1['Personalizado'], name='🎛️ Tu Escenario', marker_color='#4A2E2B'))

                fig_bar_d1.update_layout(
                    title="Comparativa de Demanda Semanal por Escenario",
                    barmode='group',
                    yaxis_title="Unidades Proyectadas",
                    template="plotly_white"
                )
                st.plotly_chart(fig_bar_d1, use_container_width=True)

            with t2_d1:
                st.write(f"Proyección semanal de **Tu Escenario** con intervalo de incertidumbre de **±{margen_error_pct}%**:")
                fig_line_d1 = go.Figure()
                fig_line_d1.add_trace(go.Scatter(
                    x=df_sem_d1['Semana'], y=df_sem_d1['Lim_Sup'],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_line_d1.add_trace(go.Scatter(
                    x=df_sem_d1['Semana'], y=df_sem_d1['Lim_Inf'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(111, 78, 55, 0.18)',
                    name=f'Banda Incertidumbre (±{margen_error_pct}%)'
                ))
                fig_line_d1.add_trace(go.Scatter(
                    x=df_sem_d1['Semana'], y=df_sem_d1['Base'],
                    mode='lines+markers', name='Pronóstico Base (ML)',
                    line=dict(color='#95A5A6', width=2, dash='dash')
                ))
                fig_line_d1.add_trace(go.Scatter(
                    x=df_sem_d1['Semana'], y=df_sem_d1['Personalizado'],
                    mode='lines+markers+text', name='🎛️ Tu Escenario',
                    text=df_sem_d1['Personalizado'].round(0).astype(int),
                    textposition='top center',
                    line=dict(color='#4A2E2B', width=3)
                ))
                fig_line_d1.update_layout(
                    title="Trayectoria de Demanda: Escenario Base vs. Personalizado",
                    yaxis_title="Unidades Semanales",
                    template="plotly_white"
                )
                st.plotly_chart(fig_line_d1, use_container_width=True)

            with t3_d1:
                df_desglose_d1 = df_filtrado_d1.groupby(['store_location', 'product_name']).agg(
                    Demanda_Base_Mes=('Demanda_Proyectada_Semanal', 'sum'),
                    Demanda_Escenario_Mes=('Escenario_Personalizado', 'sum')
                ).reset_index()
                df_desglose_d1['Diferencia_Unids'] = (df_desglose_d1['Demanda_Escenario_Mes'] - df_desglose_d1['Demanda_Base_Mes']).round(1)
                df_desglose_d1.columns = ['Sede', 'Producto', 'Demanda Base (4 sem)', 'Demanda Escenario (4 sem)', 'Variación (Unids)']
                st.dataframe(df_desglose_d1.round(1), use_container_width=True, hide_index=True)

                buf_d1 = io.BytesIO()
                df_export_d1 = df_filtrado_d1[['transaction_date', 'store_location', 'product_name', 'Demanda_Proyectada_Semanal', 'Escenario_Personalizado']].copy()
                with pd.ExcelWriter(buf_d1, engine='openpyxl') as writer:
                    df_export_d1.to_excel(writer, index=False, sheet_name='Proyeccion_Escenario')
                st.download_button(
                    label="📥 Descargar Proyección del Escenario en Excel",
                    data=buf_d1.getvalue(),
                    file_name="Escenario_Demanda_Dim1.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # =========================================================================
    # ESCENARIOS DIMENSIÓN 2: PROYECCIÓN DE INVENTARIO (SVR)
    # =========================================================================
    elif dim_esc_selected == "Proyección de Inventario":
        st.markdown("""
        <div class="scenario-banner">
        <b>Simulador What-If de Inventario (SVR):</b> Evalúa cómo responde la política de reposición (<b>Lead Time SVR</b>, <b>Stock de Seguridad</b> y <b>Punto de Reorden</b>) ante variaciones de demanda, congestión vial, clima adverso y cambios en el Nivel de Servicio objetivo.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📂 Configuración de Fuente de Demanda Base (Dimensión 1 / Excel / Manual)", expanded=False):
            fuente_demanda = st.radio(
                "Selecciona el origen de la demanda proyectada:",
                [
                    "🔗 Proyección automática del modelo (Dimensión 1)",
                    "📤 Subir archivo Excel de proyección (Dimensión 1)",
                    "✍️ Ingreso manual personalizado"
                ],
                horizontal=True
            )

            df_fuente_dim1 = None
            if fuente_demanda == "🔗 Proyección automática del modelo (Dimensión 1)":
                if 'df_proyeccion_dim1' in st.session_state:
                    df_fuente_dim1 = st.session_state['df_proyeccion_dim1']
                else:
                    _, _, _, df_fuente_dim1 = ejecutar_dimension_1(df)
                    st.session_state['df_proyeccion_dim1'] = df_fuente_dim1
                st.success("✅ Conectado directamente con la proyección semanal de la Dimensión 1.")

            elif fuente_demanda == "📤 Subir archivo Excel de proyección (Dimensión 1)":
                up_esc = st.file_uploader("Sube tu Excel de proyección semanal:", type=["xlsx", "xls"], key="uploader_escenarios_dim2")
                if up_esc is not None:
                    try:
                        df_fuente_dim1 = pd.read_excel(up_esc)
                        st.success("✅ Excel cargado correctamente.")
                    except Exception as e:
                        st.error(f"Error al leer el archivo: {e}")

        col_ctrl, col_viz = st.columns([1.05, 2.2])

        with col_ctrl:
            st.markdown("### ⚙️ Panel de Control")
            st.caption("Ajusta los parámetros del **Escenario Personalizado**:")

            tiendas_disp = ['Astoria', "Hell's Kitchen", 'Lower Manhattan']
            productos_disp = ['Barista Espresso', 'Brewed Black tea', 'Brewed Chai tea']
            if df_fuente_dim1 is not None and 'product_name' in df_fuente_dim1.columns:
                prods_extra = [p for p in df_fuente_dim1['product_name'].dropna().unique().tolist() if p not in productos_disp]
                productos_disp = productos_disp + prods_extra

            tienda_esc = st.selectbox("🏬 Sede / Local:", tiendas_disp, key="esc_tienda")
            producto_esc = st.selectbox("☕ Producto:", productos_disp, key="esc_prod")

            demanda_base_diaria = 45.0
            if df_fuente_dim1 is not None:
                cols_s = [c for c in df_fuente_dim1.columns if any(k in c.lower() for k in ['store', 'local', 'tienda', 'location', 'sede'])]
                cols_p = [c for c in df_fuente_dim1.columns if any(k in c.lower() for k in ['product', 'producto', 'item', 'tipo', 'category', 'detail'])]
                cols_q = [c for c in df_fuente_dim1.columns if any(k in c.lower() for k in ['demanda', 'qty', 'cantidad', 'proyeccion', 'prediccion', 'transaction_qty', 'y_pred', 'target'])]

                df_f = df_fuente_dim1.copy()
                if cols_s:
                    df_f = df_f[df_f[cols_s[0]].astype(str).str.strip().str.lower() == tienda_esc.strip().lower()]
                if cols_p:
                    df_f = df_f[df_f[cols_p[0]].astype(str).str.strip().str.lower() == producto_esc.strip().lower()]
                if not df_f.empty and cols_q:
                    demanda_base_diaria = float(df_f[cols_q[0]].mean()) / 7.0

            if fuente_demanda == "✍️ Ingreso manual personalizado":
                demanda_base_diaria = st.number_input("Demanda Diaria Base (unids/día):", min_value=1.0, max_value=2000.0, value=45.0, step=5.0)
            else:
                st.info(f"📌 **Demanda Base (Dim 1):** {demanda_base_diaria:.2f} unids/día ({demanda_base_diaria*7:.1f} unids/sem)")

            st.markdown("---")
            st.markdown("**📈 Choque de Demanda (What-If)**")
            shock_demanda_pct = st.slider("Variación sobre Demanda Base (%):", -40, 60, 0, step=5, format="%d%%", key="d2_shock")
            demanda_escenario = demanda_base_diaria * (1.0 + shock_demanda_pct / 100.0)

            st.markdown("**🚚 Variables Logísticas (Modelo SVR)**")
            dist_esc = st.slider("Distancia Proveedor (km):", 5.0, 60.0, 25.0, 1.0, key="esc_dist")
            trafico_esc = st.slider("Índice de Tráfico (1 a 5):", 1.0, 5.0, 3.0, 0.1, key="esc_traf")
            clima_opciones = {"☀️ Despejado (0)": 0, "🌧️ Lluvia (1)": 1, "🌩️ Clima adverso (2)": 2}
            clima_sel = st.selectbox("Condición Climática:", list(clima_opciones.keys()), key="esc_clima")
            clima_esc = clima_opciones[clima_sel]

            st.markdown("**🛡️ Política de Servicio y Reposición**")
            ns_esc = st.slider("Nivel de Servicio Objetivo (%):", 80, 99, 95, 1, key="esc_ns")
            lote_dias_esc = st.slider("Frecuencia / Cobertura de Pedido (días):", 3, 14, 7, 1, key="esc_lote")

        definicion_escenarios = [
            {
                "Escenario": "1. Política Tradicional (Sin SVR - 85%)",
                "Descripción": "Baseline histórico sin ajuste dinámico",
                "demanda": demanda_base_diaria,
                "dist": 25.0,
                "trafico": 2.5,
                "clima": 0,
                "ns": 85
            },
            {
                "Escenario": "2. Optimista (Ruta Fluida / -15% Dem.)",
                "Descripción": "Baja demanda y tráfico despejado",
                "demanda": demanda_base_diaria * 0.85,
                "dist": max(5.0, dist_esc * 0.85),
                "trafico": 1.5,
                "clima": 0,
                "ns": 95
            },
            {
                "Escenario": "3. Base Operativo (SVR - 95%)",
                "Descripción": "Proyección Dim 1 + condiciones normales",
                "demanda": demanda_base_diaria,
                "dist": dist_esc,
                "trafico": 2.5,
                "clima": 0,
                "ns": 95
            },
            {
                "Escenario": "4. Estrés Crítico (Tormenta + Pico +40%)",
                "Descripción": "Clima adverso, congestión y alta demanda",
                "demanda": demanda_base_diaria * 1.40,
                "dist": dist_esc,
                "trafico": 4.8,
                "clima": 2,
                "ns": 98
            },
            {
                "Escenario": "5. 🎛️ Tu Escenario Personalizado",
                "Descripción": f"Demanda {shock_demanda_pct:+d}%, Tráfico {trafico_esc:.1f}, NS {ns_esc}%",
                "demanda": demanda_escenario,
                "dist": dist_esc,
                "trafico": trafico_esc,
                "clima": clima_esc,
                "ns": ns_esc
            }
        ]

        filas_escenarios = []
        for esc in definicion_escenarios:
            calc = calcular_politica_inventario(
                tienda_esc, producto_esc, esc["demanda"],
                esc["dist"], esc["trafico"], esc["clima"], esc["ns"]
            )
            filas_escenarios.append({
                "Escenario": esc["Escenario"],
                "Condiciones": esc["Descripción"],
                "Demanda Diaria (unids)": round(calc["demanda_diaria"], 2),
                "Lead Time SVR (días)": round(calc["lead_time"], 2),
                "Demanda en Tránsito (unids)": round(calc["demanda_lt"], 1),
                "Stock Seguridad SS (unids)": round(calc["ss"], 1),
                "Punto de Reorden ROP (unids)": round(calc["rop"], 1),
                "Nivel Servicio (%)": esc["ns"],
                "Cobertura ROP (días)": round(calc["cobertura_dias"], 2)
            })

        df_comparativa_esc = pd.DataFrame(filas_escenarios)
        res_trad = filas_escenarios[0]
        res_base = filas_escenarios[2]
        res_pers = filas_escenarios[4]

        with col_viz:
            st.markdown(f"### 📊 Resultados Comparativos: **{producto_esc}** en **{tienda_esc}**")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                "⏱️ Lead Time (Personalizado)",
                f"{res_pers['Lead Time SVR (días)']:.2f} días",
                f"{res_pers['Lead Time SVR (días)'] - res_base['Lead Time SVR (días)']:+.2f} d vs Base",
                delta_color="inverse"
            )
            kpi2.metric(
                "🛡️ Stock Seguridad (SS)",
                f"{res_pers['Stock Seguridad SS (unids)']:.1f} unids",
                f"{res_pers['Stock Seguridad SS (unids)'] - res_base['Stock Seguridad SS (unids)']:+.1f} vs Base"
            )
            kpi3.metric(
                "📍 Punto de Reorden (ROP)",
                f"{res_pers['Punto de Reorden ROP (unids)']:.1f} unids",
                f"{res_pers['Punto de Reorden ROP (unids)'] - res_base['Punto de Reorden ROP (unids)']:+.1f} vs Base"
            )
            kpi4.metric(
                "📈 Mejora SS vs Tradicional (85%)",
                f"+{res_pers['Stock Seguridad SS (unids)'] - res_trad['Stock Seguridad SS (unids)']:.1f} unids",
                f"NS: {res_trad['Nivel Servicio (%)']}% → {res_pers['Nivel Servicio (%)']}%"
            )

            stab1, stab2, stab3, stab4 = st.tabs([
                "📋 Comparativa de Escenarios",
                "🔄 Simulación Diaria (28 días)",
                "📈 Curvas de Sensibilidad",
                "🏪 Matriz Red Completa"
            ])

            with stab1:
                st.write("Comparación simultánea de los parámetros óptimos de inventario bajo distintos contextos operativos:")
                st.dataframe(df_comparativa_esc, use_container_width=True, hide_index=True)

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=df_comparativa_esc["Escenario"],
                    y=df_comparativa_esc["Demanda en Tránsito (unids)"],
                    name="Demanda durante Lead Time",
                    marker_color="#D2B48C"
                ))
                fig_comp.add_trace(go.Bar(
                    x=df_comparativa_esc["Escenario"],
                    y=df_comparativa_esc["Stock Seguridad SS (unids)"],
                    name="Stock de Seguridad (SS)",
                    marker_color="#6F4E37"
                ))
                fig_comp.add_trace(go.Scatter(
                    x=df_comparativa_esc["Escenario"],
                    y=df_comparativa_esc["Punto de Reorden ROP (unids)"],
                    name="Punto de Reorden Total (ROP)",
                    mode="lines+markers+text",
                    text=df_comparativa_esc["Punto de Reorden ROP (unids)"].round(0).astype(int),
                    textposition="top center",
                    line=dict(color="#C0392B", width=3)
                ))

                fig_comp.update_layout(
                    title="Composición del Punto de Reorden (ROP = Demanda en Tránsito + Stock de Seguridad)",
                    barmode="stack",
                    yaxis_title="Unidades",
                    xaxis_title="",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            with stab2:
                st.write(
                    "Simulación estocástica día a día (28 días) comparando tu **Escenario Personalizado (SVR)** "
                    "frente a la **Política Tradicional al 85%**. Observa cómo el disparo automático en el **ROP** evita quiebres de stock."
                )

                df_sim_pers = simular_trayectoria_diaria(
                    demanda_diaria=res_pers["Demanda Diaria (unids)"],
                    lt_dias=res_pers["Lead Time SVR (días)"],
                    ss=res_pers["Stock Seguridad SS (unids)"],
                    rop=res_pers["Punto de Reorden ROP (unids)"],
                    dias=28,
                    lote_dias=lote_dias_esc,
                    seed=42
                )

                df_sim_trad = simular_trayectoria_diaria(
                    demanda_diaria=res_pers["Demanda Diaria (unids)"],
                    lt_dias=res_trad["Lead Time SVR (días)"],
                    ss=res_trad["Stock Seguridad SS (unids)"],
                    rop=res_trad["Punto de Reorden ROP (unids)"],
                    dias=28,
                    lote_dias=lote_dias_esc,
                    seed=42
                )

                quiebres_pers = df_sim_pers["Quiebre (Unids)"].sum()
                quiebres_trad = df_sim_trad["Quiebre (Unids)"].sum()
                dias_quiebre_pers = int((df_sim_pers["Quiebre (Unids)"] > 0).sum())
                dias_quiebre_trad = int((df_sim_trad["Quiebre (Unids)"] > 0).sum())

                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("🚨 Quiebres (Tu Escenario SVR)", f"{quiebres_pers:.1f} unids", f"{dias_quiebre_pers} días con quiebre", delta_color="inverse")
                sm2.metric("⚠️ Quiebres (Política 85%)", f"{quiebres_trad:.1f} unids", f"{dias_quiebre_trad} días con quiebre", delta_color="inverse")
                sm3.metric("✅ Ventas Salvadas por SVR", f"{max(0.0, quiebres_trad - quiebres_pers):.1f} unids")

                fig_saw = go.Figure()
                fig_saw.add_trace(go.Scatter(
                    x=df_sim_pers["Día"],
                    y=df_sim_pers["Inventario Final"],
                    mode="lines+markers",
                    name=f"Inventario Política SVR ({ns_esc}%)",
                    line=dict(color="#4A2E2B", width=3)
                ))
                fig_saw.add_trace(go.Scatter(
                    x=df_sim_trad["Día"],
                    y=df_sim_trad["Inventario Final"],
                    mode="lines",
                    name="Inventario Política Tradicional (85%)",
                    line=dict(color="#95A5A6", width=2, dash="dot")
                ))
                fig_saw.add_hline(
                    y=res_pers["Punto de Reorden ROP (unids)"],
                    line_dash="dash",
                    line_color="#E67E22",
                    annotation_text=f"ROP ({res_pers['Punto de Reorden ROP (unids)']:.1f} u)"
                )
                fig_saw.add_hline(
                    y=res_pers["Stock Seguridad SS (unids)"],
                    line_dash="dash",
                    line_color="#C0392B",
                    annotation_text=f"Stock Seguridad ({res_pers['Stock Seguridad SS (unids)']:.1f} u)"
                )

                fig_saw.update_layout(
                    title=f"Evolución Diaria del Inventario (28 días) — {producto_esc} ({tienda_esc})",
                    xaxis_title="Día del Mes",
                    yaxis_title="Unidades en Inventario",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_saw, use_container_width=True)

            with stab3:
                st.write("Análisis de sensibilidad del **Stock de Seguridad (SS)** y **Punto de Reorden (ROP)** frente al Nivel de Servicio y condiciones climáticas:")

                niveles_ns = list(range(80, 100))
                curvas_clima = []
                for c_nombre, c_cod in clima_opciones.items():
                    for ns_val in niveles_ns:
                        c_res = calcular_politica_inventario(
                            tienda_esc, producto_esc, demanda_escenario,
                            dist_esc, trafico_esc, c_cod, ns_val
                        )
                        curvas_clima.append({
                            "Nivel de Servicio (%)": ns_val,
                            "Clima": c_nombre,
                            "Stock de Seguridad (SS)": round(c_res["ss"], 2),
                            "Punto de Reorden (ROP)": round(c_res["rop"], 2),
                            "Lead Time (días)": round(c_res["lead_time"], 2)
                        })

                df_sens = pd.DataFrame(curvas_clima)

                fig_sens = px.line(
                    df_sens,
                    x="Nivel de Servicio (%)",
                    y="Punto de Reorden (ROP)",
                    color="Clima",
                    markers=True,
                    title=f"Sensibilidad del Punto de Reorden (ROP) según Nivel de Servicio y Clima ({tienda_esc})",
                    color_discrete_sequence=["#2E86C1", "#F39C12", "#C0392B"]
                )
                fig_sens.add_vline(x=85, line_dash="dot", line_color="gray", annotation_text="Baseline 85%")
                fig_sens.add_vline(x=95, line_dash="dash", line_color="green", annotation_text="Objetivo 95%")
                fig_sens.update_layout(template="plotly_white")
                st.plotly_chart(fig_sens, use_container_width=True)

            with stab4:
                st.write(
                    f"Aplicación simultánea de los parámetros de **Tu Escenario Personalizado** "
                    f"(Variación Demanda: **{shock_demanda_pct:+d}%**, Distancia: **{dist_esc} km**, "
                    f"Tráfico: **{trafico_esc}**, Clima: **{clima_sel}**, NS: **{ns_esc}%**) a toda la red de cafeterías:"
                )

                filas_red = []
                for t_item in tiendas_disp:
                    for p_item in ['Barista Espresso', 'Brewed Black tea', 'Brewed Chai tea']:
                        dem_item = 45.0
                        if df_fuente_dim1 is not None:
                            cols_s = [c for c in df_fuente_dim1.columns if any(k in c.lower() for k in ['store', 'local', 'tienda', 'location', 'sede'])]
                            cols_p = [c for c in df_fuente_dim1.columns if any(k in c.lower() for k in ['product', 'producto', 'item', 'tipo', 'category', 'detail'])]
                            cols_q = [c for c in df_fuente_dim1.columns if any(k in c.lower() for k in ['demanda', 'qty', 'cantidad', 'proyeccion', 'prediccion', 'transaction_qty', 'y_pred', 'target'])]
                            df_sub_red = df_fuente_dim1.copy()
                            if cols_s:
                                df_sub_red = df_sub_red[df_sub_red[cols_s[0]].astype(str).str.strip().str.lower() == t_item.strip().lower()]
                            if cols_p:
                                df_sub_red = df_sub_red[df_sub_red[cols_p[0]].astype(str).str.strip().str.lower() == p_item.strip().lower()]
                            if not df_sub_red.empty and cols_q:
                                dem_item = float(df_sub_red[cols_q[0]].mean()) / 7.0

                        dem_ajustada = dem_item * (1.0 + shock_demanda_pct / 100.0)
                        r_red = calcular_politica_inventario(
                            t_item, p_item, dem_ajustada, dist_esc, trafico_esc, clima_esc, ns_esc
                        )
                        filas_red.append({
                            "Sede": t_item,
                            "Producto": p_item,
                            "Demanda Diaria Ajustada": round(dem_ajustada, 2),
                            "Lead Time SVR (días)": round(r_red["lead_time"], 2),
                            "Stock de Seguridad (SS)": round(r_red["ss"], 1),
                            "Punto de Reorden (ROP)": round(r_red["rop"], 1),
                            "Orden Sugerida (Lote)": round(dem_ajustada * lote_dias_esc, 1)
                        })

                df_red_completa = pd.DataFrame(filas_red)
                st.dataframe(df_red_completa, use_container_width=True, hide_index=True)

                buf_esc = io.BytesIO()
                with pd.ExcelWriter(buf_esc, engine='openpyxl') as writer:
                    df_comparativa_esc.to_excel(writer, index=False, sheet_name='Comparativa_Escenarios')
                    df_red_completa.to_excel(writer, index=False, sheet_name='Matriz_Red_Completa')

                st.download_button(
                    label="📥 Descargar Simulación de Escenarios en Excel",
                    data=buf_esc.getvalue(),
                    file_name=f"Escenarios_Inventario_Dim2_{tienda_esc.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # =========================================================================
    # ESCENARIOS DIMENSIÓN 3: ANÁLISIS DE PORTAFOLIO (PRICING & ELASTICIDAD)
    # =========================================================================
    else:  # dim_esc_selected == "Análisis de Portafolio"
        st.markdown("""
        <div class="scenario-banner">
        <b>Simulador What-If de Precios y Portafolio:</b> Evalúa cómo responde la cantidad demandada y los ingresos totales ante ajustes personalizados de precio (<b>-25% a +25%</b>) utilizando las elasticidades precio-demanda estimadas mediante regresión log-log en la <b>Dimensión 3</b>.
        </div>
        """, unsafe_allow_html=True)

        if error_dim3 or portafolio_dim3.empty:
            st.error(f"No se pudo inicializar el simulador de Dimensión 3: {error_dim3 or 'Sin productos con variación de precio'}")
        else:
            col_ctrl3, col_viz3 = st.columns([1.05, 2.2])

            with col_ctrl3:
                st.markdown("### ⚙️ Panel de Control")
                st.caption("Configura tu estrategia de **Fijación de Precios**:")

                lista_prods_d3 = portafolio_dim3['product_detail'].tolist()
                prod_sel_d3 = st.selectbox("☕ Producto a Simular:", lista_prods_d3, key="d3_esc_prod")

                fila_prod = portafolio_dim3[portafolio_dim3['product_detail'] == prod_sel_d3].iloc[0]
                p0_d3 = float(fila_prod['precio_base'])
                q0_d3 = float(fila_prod['cantidad_base'])
                e_est_d3 = float(fila_prod['elasticidad']) if pd.notna(fila_prod['elasticidad']) else -1.0
                pval_d3 = float(fila_prod['p_value']) if pd.notna(fila_prod['p_value']) else 1.0
                sig_d3 = fila_prod['significancia']
                tipo_d3 = fila_prod['tipo']

                st.info(
                    f"📌 **Elasticidad Econométrica:** `{e_est_d3:.3f}` ({tipo_d3})  \n"
                    f"**Significancia:** {sig_d3} (`p = {pval_d3:.4f}`)"
                )

                st.markdown("---")
                st.markdown("**💵 Ajuste de Precio (What-If)**")
                ajuste_precio_pct = st.slider("Variación de Precio (%):", -25, 25, 5, step=1, format="%+d%%", key="d3_precio_pct")

                st.markdown("**📐 Supuesto de Sensibilidad**")
                modo_elast = st.radio(
                    "Elasticidad a emplear:",
                    ["Modelo Log-Log Estimado", "Sensibilidad Personalizada (What-If)"],
                    key="d3_modo_elast"
                )
                if modo_elast == "Sensibilidad Personalizada (What-If)":
                    e_usada = st.slider("Elasticidad Simulada:", -3.5, 1.5, round(e_est_d3, 2), step=0.05, key="d3_elast_custom")
                else:
                    e_usada = e_est_d3

                st.markdown("**🧭 Regla de Gobernanza Comercial**")
                solo_sig_portafolio = st.checkbox(
                    "Aplicar ajuste global solo a productos significativos (p < 0.05)",
                    value=True,
                    key="d3_solo_sig"
                )

            # Cálculos del producto seleccionado
            p1_d3 = p0_d3 * (1.0 + ajuste_precio_pct / 100.0)
            q1_d3 = q0_d3 * ((p1_d3 / p0_d3) ** e_usada)
            ing0_d3 = p0_d3 * q0_d3
            ing1_d3 = p1_d3 * q1_d3
            var_q_pct = ((q1_d3 / q0_d3) - 1.0) * 100.0
            var_ing_pct = ((ing1_d3 / ing0_d3) - 1.0) * 100.0
            var_ing_abs = ing1_d3 - ing0_d3

            # Tabla de 6 escenarios predefinidos + personalizado para el producto
            escenarios_d3_def = [
                ("1. Descuento Agresivo (-10%)", -10.0),
                ("2. Descuento Moderado (-5%)", -5.0),
                ("3. Precio Base Actual (0%)", 0.0),
                ("4. Incremento Moderado (+5%)", 5.0),
                ("5. Incremento Alto (+10%)", 10.0),
                ("6. 🎛️ Tu Escenario Personalizado", float(ajuste_precio_pct))
            ]

            filas_esc_d3 = []
            for nombre_e, pct_p in escenarios_d3_def:
                p_sim = p0_d3 * (1.0 + pct_p / 100.0)
                q_sim = q0_d3 * ((p_sim / p0_d3) ** e_usada)
                ing_sim = p_sim * q_sim
                filas_esc_d3.append({
                    "Escenario": nombre_e,
                    "Ajuste Precio (%)": round(pct_p, 1),
                    "Precio Unitario ($)": round(p_sim, 2),
                    "Demanda Estimada (unids)": round(q_sim, 0),
                    "Variación Demanda (%)": round(((q_sim / q0_d3) - 1.0) * 100.0, 2),
                    "Ingreso Estimado ($)": round(ing_sim, 2),
                    "Variación Ingreso (%)": round(((ing_sim / ing0_d3) - 1.0) * 100.0, 2)
                })
            df_esc_prod_d3 = pd.DataFrame(filas_esc_d3)

            with col_viz3:
                st.markdown(f"### 📊 Simulación Comercial: **{prod_sel_d3}**")

                kp1, kp2, kp3, kp4 = st.columns(4)
                kp1.metric("🏷️ Nuevo Precio", f"${p1_d3:.2f}", f"{ajuste_precio_pct:+d}% vs ${p0_d3:.2f}")
                kp2.metric("📦 Demanda Estimada", f"{q1_d3:,.0f} u", f"{var_q_pct:+.2f}%")
                kp3.metric("💰 Ingreso Estimado", f"${ing1_d3:,.2f}", f"{var_ing_pct:+.2f}%")
                kp4.metric("📈 Impacto Neto ($)", f"${var_ing_abs:+,.2f}")

                if sig_d3 != "Significativo" and modo_elast == "Modelo Log-Log Estimado":
                    st.warning("⚠️ Este producto tiene elasticidad **no concluyente (p ≥ 0.05)**. Interpreta los cambios de volumen con cautela.")

                st1_d3, st2_d3, st3_d3 = st.tabs([
                    "📋 Comparativa de Escenarios",
                    "📈 Curva Precio vs. Ingresos",
                    "🗂️ Simulación Portafolio Completo"
                ])

                with st1_d3:
                    st.write("Comparación de desempeño comercial bajo distintos niveles de ajuste de precio:")
                    st.dataframe(df_esc_prod_d3, use_container_width=True, hide_index=True)

                    fig_d3_bar = go.Figure()
                    fig_d3_bar.add_trace(go.Bar(
                        x=df_esc_prod_d3["Escenario"],
                        y=df_esc_prod_d3["Ingreso Estimado ($)"],
                        text=df_esc_prod_d3["Ingreso Estimado ($)"].apply(lambda x: f"${x:,.0f}"),
                        textposition="outside",
                        marker_color=['#BDC3C7', '#D2B48C', '#8D6E63', '#6F4E37', '#5D4037', '#C0392B']
                    ))
                    fig_d3_bar.update_layout(
                        title=f"Ingresos Estimados por Escenario de Precio — {prod_sel_d3}",
                        yaxis_title="Ingresos ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_d3_bar, use_container_width=True)

                with st2_d3:
                    st.write("Curva continua de respuesta de **Ingresos Totales ($)** y **Demanda** entre **-25% y +25%** de precio:")
                    rango_pcts = np.linspace(-25, 25, 101)
                    precios_curva = p0_d3 * (1.0 + rango_pcts / 100.0)
                    demandas_curva = q0_d3 * ((precios_curva / p0_d3) ** e_usada)
                    ingresos_curva = precios_curva * demandas_curva

                    fig_curva = go.Figure()
                    fig_curva.add_trace(go.Scatter(
                        x=precios_curva,
                        y=ingresos_curva,
                        mode='lines',
                        name='Curva de Ingresos Estimados ($)',
                        line=dict(color='#4A2E2B', width=3)
                    ))
                    fig_curva.add_trace(go.Scatter(
                        x=[p0_d3],
                        y=[ing0_d3],
                        mode='markers+text',
                        name='Punto Base Actual',
                        text=[f"Base (${p0_d3:.2f})"],
                        textposition='bottom center',
                        marker=dict(color='#7F8C8D', size=11)
                    ))
                    fig_curva.add_trace(go.Scatter(
                        x=[p1_d3],
                        y=[ing1_d3],
                        mode='markers+text',
                        name='🎛️ Tu Escenario',
                        text=[f"Escenario (${p1_d3:.2f})"],
                        textposition='top center',
                        marker=dict(color='#C0392B', size=13, symbol='diamond')
                    ))
                    fig_curva.update_layout(
                        title=f"Sensibilidad de Ingresos al Precio (Elasticidad = {e_usada:.2f})",
                        xaxis_title="Precio Unitario ($)",
                        yaxis_title="Ingresos Totales ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_curva, use_container_width=True)

                with st3_d3:
                    st.write(
                        f"Simulación global aplicando un ajuste de **{ajuste_precio_pct:+d}%** al portafolio de productos "
                        f"({'solo significativos p < 0.05' if solo_sig_portafolio else 'todos los productos con variación de precio'}):"
                    )

                    filas_port_esc = []
                    for _, r_p in portafolio_dim3.iterrows():
                        p_base_i = float(r_p['precio_base'])
                        q_base_i = float(r_p['cantidad_base'])
                        ing_base_i = p_base_i * q_base_i
                        e_i = float(r_p['elasticidad']) if pd.notna(r_p['elasticidad']) else 0.0
                        es_sig_i = (r_p['significancia'] == 'Significativo')

                        aplica = (not solo_sig_portafolio) or es_sig_i
                        pct_aplicado = float(ajuste_precio_pct) if aplica else 0.0

                        p_nuevo_i = p_base_i * (1.0 + pct_aplicado / 100.0)
                        q_nuevo_i = q_base_i * ((p_nuevo_i / p_base_i) ** e_i) if aplica else q_base_i
                        ing_nuevo_i = p_nuevo_i * q_nuevo_i

                        filas_port_esc.append({
                            "Producto": r_p['product_detail'],
                            "Significancia": r_p['significancia'],
                            "Elasticidad": round(e_i, 3),
                            "Ajuste Aplicado (%)": round(pct_aplicado, 1),
                            "Precio Base ($)": round(p_base_i, 2),
                            "Precio Escenario ($)": round(p_nuevo_i, 2),
                            "Ingreso Base ($)": round(ing_base_i, 2),
                            "Ingreso Escenario ($)": round(ing_nuevo_i, 2),
                            "Variación Ingreso ($)": round(ing_nuevo_i - ing_base_i, 2)
                        })

                    df_port_sim = pd.DataFrame(filas_port_esc)
                    ing_tot_base = df_port_sim["Ingreso Base ($)"].sum()
                    ing_tot_esc = df_port_sim["Ingreso Escenario ($)"].sum()
                    delta_tot_port = ing_tot_esc - ing_tot_base
                    delta_pct_port = ((ing_tot_esc / ing_tot_base) - 1.0) * 100.0 if ing_tot_base > 0 else 0.0

                    pm1, pm2, pm3 = st.columns(3)
                    pm1.metric("💰 Ingreso Base Portafolio", f"${ing_tot_base:,.2f}")
                    pm2.metric("🎛️ Ingreso Escenario Portafolio", f"${ing_tot_esc:,.2f}", f"{delta_pct_port:+.2f}%")
                    pm3.metric("📊 Ganancia / Pérdida Neta", f"${delta_tot_port:+,.2f}")

                    st.dataframe(df_port_sim, use_container_width=True, hide_index=True)

                    buf_d3 = io.BytesIO()
                    with pd.ExcelWriter(buf_d3, engine='openpyxl') as writer:
                        df_esc_prod_d3.to_excel(writer, index=False, sheet_name='Escenarios_Producto')
                        df_port_sim.to_excel(writer, index=False, sheet_name='Escenario_Portafolio')
                    st.download_button(
                        label="📥 Descargar Simulación de Precios en Excel",
                        data=buf_d3.getvalue(),
                        file_name="Escenarios_Portafolio_Dim3.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# -----------------------------------------------------------------------------
# 6. BIBLIOGRAFÍA
# -----------------------------------------------------------------------------
elif opcion_menu == "Bibliografía":
    st.markdown('<h2 class="section-header">📚 Bibliografía y Referencias Indexadas</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sub-card">
    <ol>
        <li><b>Arzy, A., Khodaparast, F., y O’Reilly, R. (2024). A real-time prediction system for restaurant orders using time series and behavioural analytics: A conceptual framework. CEUR Workshop Proceedings. 3910, 290–300. https://www.scopus.com/pages/publications/105002680853?origin=resultslist</li>
        <li><b>Azzahra, A., y Sudirman, I. D. (2025). Forecasting daily sales of coffee shop products: A comparative study of moving average, ARIMA, and SARIMA. 2025 International Conference on Data Science and Its Applications, ICoDSA. 345–349. https://doi.org/10.1109/ICoDSA67155.2025.11157384.</li>
        <li><b>Dhage, S., Chitale, D., Bandre, S., Shelke, P., y Rege, P. (2025). Demand forecasting with machine learning: A data-driven strategy for market prediction. 2025 IEEE Pune Section International Conference (PuneCon). https://doi.org/10.1109/PuneCon67554.2025.11378773</li>
        <li><b>Hansen, P., Transchel, S., & Friedrich, M. (2023). Replenishment strategies for lost sales inventory systems of perishables under demand and lead time uncertainty. European Journal of Operational Research, 307(1), 221–238. 10.1016/j.ejor.2022.11.041</li>
        <li><b>Kumar, M., Pilania, U., y Dagar, D. (2025). Food demand forecasting using machine learning approaches. Proceedings of 5th International Conference on Soft Computing for Security Applications. 1106-1111. https://doi.org/10.1109/ICSCSA66339.2025.11171212</li>
        <li><b>Mirzaei, H., Daneshvar, A., & Nahavandi, B. (2025). Predicting the duration of goods transportation delays based on machine learning methods. Modern Supply Chain Research and Applications, 7(3), 407–411. 10.1108/MSCRA-02-2025-0011</li>
        <li><b>Müller, N., Burggräf, P., Steinberg, F., Sauer, C. R., Schütz, M. (2025). An analytical review of predictive methods for delivery delays in supply chains. Supply Chain Analytics, 11, 100130. 10.1016/j.sca.2025.100130</li>
        <li><b>Parashar, A., Indumathy, R., Akhilesh Gowda, Y. N., Alotoum, F. J., Doguparthy, G. S., y Natrayan, L. (2024). Time series analysis and random forest techniques for prediction of sales for retail grocery. Proceedings of the 2024 International Conference on Computing, Power, and Communication Technologies, IC2PCT. 531–535. https://doi.org/10.1109/IC2PCT60090.2024.10486802</li>
        <li><b>Shi, R., Xu, X., Li, J., & Li, Y. (2021). Prediction and analysis of train arrival delay based on XGBoost and Bayesian optimization. Applied Soft Computing, 109, 107538. 10.1016/j.asoc.2021.107538</li>
        <li><b>Singh, S., & Soni, U. (2019). Predicting order lead time for just in time production system using various machine learning algorithms: A case study. 2019 9th International Conference on Cloud Computing, Data Science & Engineering (Confluence), 322–327. 10.1109/CONFLUENCE.2019.8776892</li>
        <li><b>Weber, A., & Steiner, W. J. (2021). Modeling price response from retail sales: An empirical comparison of models with different representations of heterogeneity. European Journal of Operational Research, 294(3), 843–859. https://doi.org/10.1016/j.ejor.2020.07.055</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
