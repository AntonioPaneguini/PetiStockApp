import streamlit as st
import pandas as pd
import altair as alt
# TODO Rodar todo o modelo de transformação por aqui
# TODO Trocar a formatação dos valores de data no eixo X para o formato de mm-yy
# TODO Trocar line_chart pelo altair e formatar a linha para ficar vermelha quando ficar negativa
# TODO Adicionar tabela mostrando os valores de estoque atual, mes que o produto vai acabar, média de saida por mes e recomendação de compra ou não
# TODO filtro de lista suspensa de produtos estudados no line_chart


# Extração e Transformação de dados


# Modelagem de dados

# Transformação pre-visualização
# st.navigation(pages=["Home"])
df = pd.read_csv("datasets/estoque_futuro_estimado.csv")

df.rename(columns={"Unnamed: 0": "Datas"}, inplace=True)

# df.set_index(df.iloc[:, 0], drop=True, inplace=True)


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
            point={"filled": True, "fill": "white"},
            color=colors[i % len(colors)],
            size=2,
        )
        .encode(
            x=alt.X("Datas:T", title="Datas Futuras"),
            y=alt.Y(f"{col}:Q", title="Saldo de Estoque Especulado"),
            tooltip=[f"{col}"],
        )
        for i, col in enumerate(df_future.columns[1:])
    ]
)

zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", size=5).encode(y="y:Q")
)
chart = alt.layer(line_chart + zero_line)

st.altair_chart(chart, use_container_width=True)


st.subheader("Tabela de valores futuros", divider=True)


df = round(df)


# st.dataframe(df)
def red_background_zero_values(series):
    highlight = "background-color: red; font-weight: bold;"
    default = ""
    return [highlight if val <= 0 else default for val in series]


df_table = df[["Datas"] + prod_options].copy()
st.dataframe(
    df_table.style.apply(
        red_background_zero_values, axis=0, subset=list(df_table.columns[1:])
    )
    # df.style.highlight_between(left=-1000, right=1, color="red", axis=None)
    # df.style.highlight_min(color="red", axis=0)
    # df
)
