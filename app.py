import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    TBATS,
)
from darts.models import AutoARIMA
from darts import TimeSeries
from statsmodels.tsa.stattools import adfuller
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
import time


# Front-End Inicial
progress_bar = st.progress(
    0,
    text="Prevendo a Saída de Estoque Futura baseado nos nossos modelos. Por favor, espere...",
)
status_text = st.empty()

# TODO Organizar o codigo em funções, deixar apenas o front-end aqui
# TODO Adicionar tabela mostrando os valores de estoque atual, mes que o produto vai acabar, média de saida por mes e recomendação de compra ou não
# TODO Impedir o modelo de rodar a cada callback (fazer as transformações acontecerem via cron)
# TODO Alterar a barra de progresso

# Extração e Transformação de dados

df = pd.read_excel("datasets/quantidade_media_prod_tri.xlsx")
df["dia_do_mes"] = df["dia_do_mes"].fillna(method="ffill")
df = df.query(
    'not descricao_prod.str.contains("Kit") and not descricao_prod.str.contains("Box") and not descricao_prod.str.contains("Pasta")'
)
df["data"] = pd.to_datetime(
    {"year": df["ano"], "month": df["mês"], "day": df["dia_do_mes"]}, format="d%/m%/y%"
)
df["fornecedor"] = df["descricao_prod"].apply(
    lambda x: "GMP"
    if "Petisco Natural Sabor" in x
    else (
        "Laboratório Oriente - Suplementos"
        if "Suplemento" in x
        else ("Padaria Pet" if "BSF" in x else "Amicus")
    )
)
df_wk = pd.DataFrame(
    df.groupby([pd.Grouper(key="data", freq="W-MON"), "descricao_prod"])[
        "quantidade_total"
    ].sum()
)
df_me = pd.DataFrame(
    df.groupby([pd.Grouper(key="data", freq="ME"), "descricao_prod"])[
        "quantidade_total"
    ].sum()
)
df_wk.reset_index(inplace=True)
df_me.reset_index(inplace=True)

df_pivot = df_me.pivot_table(
    index="data", columns="descricao_prod", values="quantidade_total", fill_value=0
)

# Modelagem de dados

m_expon = ExponentialSmoothing()

m_prophet_m = Prophet(seasonality_mode="multiplicative")

m_prophet_a = Prophet(seasonality_mode="additive")

m_n_seasonal = NaiveSeasonal()

m_n_drift = NaiveDrift()

m_arima = ARIMA(0, 0, 0)

m_tbats = TBATS(season_length=4)


models = [m_expon, m_prophet_m, m_prophet_a, m_n_seasonal, m_n_drift, m_arima, m_tbats]


def eval_model(model, train, val, df_name):
    t_start = time.perf_counter()
    print("beginning: " + str(model))

    # fit the model and compute predictions
    res = model.fit(train)
    forecast = model.predict(len(val) + 6)
    try:
        # compute accuracy metrics and processing time
        res_mape = mape(val, forecast)
        res_mae = mae(val, forecast)
        res_r2 = r2_score(val, forecast)
        res_rmse = rmse(val, forecast)
        res_rmsle = rmsle(val, forecast)
        res_time = time.perf_counter() - t_start
        res_accuracy = {
            "MAPE": res_mape,
            "MAE": res_mae,
            "R squared": -res_r2,
            "RMSE": res_rmse,
            "RMSLE": res_rmsle,
            "time": res_time,
        }

        results = [forecast, res_accuracy, res]
        print("completed: " + str(model) + ":" + str(res_time) + "sec")
        model.fit(train)
        prediction = model.predict(len(val) + 6)
        plt.figure()
        series.plot(label="actual")
        prediction.plot(label="forecast", lw=3)
        plt.legend()
        plt.title(f"Previsão de {df_name} - modelo {str(model).split('())')[0]}")
    except ValueError:
        res_mae = mae(val, forecast)
        res_r2 = r2_score(val, forecast)
        res_rmse = rmse(val, forecast)
        res_rmsle = rmsle(val, forecast)
        res_time = time.perf_counter() - t_start
        res_accuracy = {
            "MAE": res_mae,
            "R squared": -res_r2,
            "RMSE": res_rmse,
            "RMSLE": res_rmsle,
            "time": res_time,
        }
        results = [forecast, res_accuracy, res]
        model.fit(train)
        prediction = model.predict(len(val) + 6)
    return results


k = 0
df_pred = pd.DataFrame()
for i in df_pivot.columns:
    df_pred_list = []
    df_ = df_pivot.iloc[:, k : k + 1]
    series = TimeSeries.from_dataframe(
        df=df_, fill_missing_dates=True, fillna_value=0, freq="ME"
    )
    train, val = series.split_before(pd.Timestamp("20250101"))
    print(f"Criando previsão de saída de Estoque para {i}")
    model_predictions = [eval_model(model, train, val, i) for model in models]
    result = adfuller(series.to_dataframe())
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))
    if result[1] <= 0.05:
        print("Reject the null hypothesis: The series is stationary")
    else:
        print("Fail to reject the null hypothesis: The series is non-stationary")
    df_acc = pd.DataFrame.from_dict(model_predictions[0][1], orient="index")
    df_acc.columns = [str(models[0])]
    for j, m in enumerate(models):
        if j > 0:
            df = pd.DataFrame.from_dict(model_predictions[j][1], orient="index")
            df.columns = [str(m)]
            df_acc = pd.concat([df_acc, df], axis=1)
        j += 1
    df_min = df_acc.idxmin(axis=1)
    for m in models:
        if str(m) in df_min.RMSE:
            forecast = m.predict(len(val) + 6)
            df_i = forecast.to_dataframe()
            if df_i.max().iloc[0] != 0:
                df_pred = pd.concat([df_pred, df_i], axis=1)
            else:
                pass
    if k >= 10:
        progress_bar.progress(99)
        status_text.text("99% Escolhendo o melhor modelo preditivo para cada produto!")
    else:
        status_text.text(f"{k * 10}% Processado, estudando o passado...")
        progress_bar.progress(k * 10)
    time.sleep(0.05)
    k += 1

progress_bar.empty()
status_text.empty()

df_estoq = pd.read_excel("datasets/quantidade_total_produtos_simples.xlsx")
df_estoq = df_estoq.iloc[:, 1:]
df_estoq.rename(columns={"saldo_fisico_produto": "Saldo Atual"}, inplace=True)
df_estoq_pivot = pd.pivot_table(
    data=df_estoq, columns="descricao_prod", values="Saldo Atual"
)
df_fut = df_pred[11:]
df_estoq_fut = pd.concat([df_estoq_pivot, -df_fut])
df_estoq_fut.dropna(axis=1, inplace=True)
df_estoq_fut = df_estoq_fut.cumsum()
df_estoq_fut.to_csv("datasets/estoque_futuro_estimado.csv")


# Transformação pre-visualização
# st.navigation(pages=["Home"])
df = pd.read_csv("datasets/estoque_futuro_estimado.csv")
#
df.rename(columns={"Unnamed: 0": "Datas"}, inplace=True)

df.set_index(df.iloc[:, 0], drop=True, inplace=True)


# Visualização de dados

st.sidebar.subheader("Selecionar Produtos")
prod_options = [
    prod_name
    for prod_name in list(df.columns[1:])
    if st.sidebar.checkbox(prod_name, True)
]

if not prod_options:
    st.warning("Selecione ao menos um produto.")
    st.stop()

df_future = df[1:]
df_future = df_future[["Datas"] + prod_options].copy()
df_future = round(df_future)

st.header("Modelo de Previsão de Estoque Futuro", divider=True)


colors = [
    "steelblue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

line_chart = alt.layer(
    *[
        alt.Chart(df_future)
        .mark_line(
            point={"filled": True, "fill": "#D55871"},
            color=colors[i % len(colors)],
            size=3,
        )
        .encode(
            x=alt.X("Datas:T", title="Datas Futuras"),
            y=alt.Y(f"{col}:Q", title="Saldo de Estoque Especulado"),
            tooltip=["Datas:T", f"{col}"],
        )
        for i, col in enumerate(df_future.columns[1:])
    ]
)

zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", size=5).encode(y="y:Q")
)
chart = alt.layer(line_chart + zero_line)
with st.container(border=True, width="stretch", height="stretch"):
    st.altair_chart(chart, use_container_width=True, height=720, width=1080)


st.subheader("Tabela de valores futuros", divider=True)


df = round(df)


# st.dataframe(df)
def red_background_zero_values(series):
    highlight = "background-color: red; font-weight: bold;"
    default = ""
    return [highlight if val <= 0 else default for val in series]


df_table = df[["Datas"] + prod_options].copy()

with st.container(border=True, width="stretch", height="stretch"):
    st.dataframe(
        df_table.style.apply(
            red_background_zero_values, axis=0, subset=list(df_table.columns[1:])
        )
        # df.style.highlight_between(left=-1000, right=1, color="red", axis=None)
        # df.style.highlight_min(color="red", axis=0)
        # df
    )
