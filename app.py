import streamlit as st
import pandas as pd

# TODO Roder todo o modelo de transformação por aqui
# TODO Trocar a formatação dos valores de data no eixo X para o formato de mm-yy
# TODO Trocar line_chart pelo altair e formatar a linha para ficar vermelha quando ficar negativa
# TODO Adicionar tabela mostrando os valores de estoque atual, mes que o produto vai acabar, média de saida por mes e recomendação de compra ou não
# TODO filtro de lista suspensa de produtos estudados no line_chart


# Extração e Transformação de dados


# Modelagem de dados

# Transformação pre-visualização
# @st.cache_data
# st.navigation(pages=["Home"])
df = pd.read_csv("datasets/estoque_futuro_estimado.csv")

df.rename(columns={"Unnamed: 0": "Datas"}, inplace=True)

# df.set_index(df.iloc[:, 0], drop=True, inplace=True)


df_future = df[1:]

# Visualização de dados


prod_options = st.sidebar.selectbox("Produtos em Estoque:", list(df_future.iloc[:, 1:]))


# "Selecione:", prod_options
st.header("Modelo de Previsão de Estoque Futuro", divider=True)

st.line_chart(
    data=round(df_future),
    x="Datas",
    y=str(df_future.columns[1]),
    y_label="Qtde Un. no Estoque",
    x_label="Datas Futuras",
)


st.subheader("Tabela de valores futuros", divider=True)


df = round(df)


# st.dataframe(df)
def red_background_zero_values(series):
    highlight = "background-color: red; font-weight: bold;"
    default = ""
    return [highlight if val <= 0 else default for val in series]


st.dataframe(
    df.style.apply(red_background_zero_values, axis=0, subset=list(df.columns[1:]))
    # df.style.highlight_between(left=-1000, right=1, color="red", axis=None)
    # df.style.highlight_min(color="red", axis=0)
    # df
)
