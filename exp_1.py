import streamlit as st
import numpy as np
import plotly.express as px
#from mealpy.evolutionary_based.GA import BaseGA
from mealpy.bio_based.BBO import OriginalBBO
#from mealpy.swarm_based.PSO import OriginalPSO as BasePSO
from mealpy.evolutionary_based.EP import OriginalEP
from opfunu.cec_based import F52017, F92017, F112017
from mealpy import FloatVar
import pandas as pd
import time

f1 = F52017(ndim=30, f_bias=0, f_shift='shift_data_2',f_matrix='M_2_D')
f2 = F92017(ndim=30, f_bias=0, f_shift='shift_data_2',f_matrix='M_2_D')
f3 = F112017(ndim=30, f_bias=0, f_shift='shift_data_2',f_matrix='M_2_D')

p1 = {
    "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
    "obj_func": f1.evaluate,
    "minmax": "min",
    "name": "F5",
    "log_to": None
}

p2 = {
    "bounds": FloatVar(lb=f2.lb, ub=f2.ub),
    "obj_func": f2.evaluate,
    "minmax": "min",
    "name": "F9",
    "log_to": None
}

p3 = {
    "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
    "obj_func": f3.evaluate,
    "minmax": "min",
    "name": "F11",
    "log_to": None
}


resultados = {
    "Problema": [],
    "unshifted": [],
    "shifted": [],
    "ratio": []
}

algoritmos = ["BBO", "EP"]

def run_algorithm(algoritmo, problem, pop_size, epoch):
    if algoritmo == "BBO":
        model = OriginalBBO(epoch=epoch, pop_size=pop_size)
    elif algoritmo == "EP":
        model = OriginalEP(epoch=epoch, pop_size=pop_size)
    else:
        raise ValueError("Algoritmo não suportado")

    model.solve(problem)

    return np.abs(model.g_best.target.fitness)

def run_experiment(algoritmo, problem, pop_size, epoch, use_shift):
    results = []
    for function_choice, problem in zip(["F5", "F9", "F11"], [p1, p2, p3]):

        best_fitness = run_algorithm(algoritmo, problem , pop_size, epoch)
        results.append(best_fitness)

        if use_shift:
            if function_choice in resultados["Problema"]:
                idx = resultados["Problema"].index(function_choice)
                if not isinstance(resultados["shifted"][idx], list):
                    resultados["shifted"][idx] = [resultados["shifted"][idx]]
                resultados["shifted"][idx].append(best_fitness)
            else:
                resultados["Problema"].append(function_choice)
                resultados["unshifted"].append(0.0)
                resultados["shifted"].append(best_fitness)
                resultados["ratio"].append(1.0)
        else:
            if function_choice in resultados["Problema"]:
                idx = resultados["Problema"].index(function_choice)
                if not isinstance(resultados["unshifted"][idx], list):
                    resultados["unshifted"][idx] = [resultados["unshifted"][idx]]
                resultados["unshifted"][idx].append(best_fitness)
            else:
                resultados["Problema"].append(function_choice)
                resultados["unshifted"].append(best_fitness)
                resultados["shifted"].append(0.0)
                resultados["ratio"].append(1.0)

def main():

    st.set_page_config(page_title="Otimização com BBO e EP", page_icon="🧊", layout='wide')

    st.title("Otimização com BBO e EP utilizando Mealpy")
    
    col1, col2 = st.columns(2)
    
    with st.sidebar:
            with st.form(key="config_form"):
                nruns = st.number_input("Número de Execuções", min_value=1, max_value=50, value=20)
                maxfes = st.number_input("Número de Avaliações de Função", min_value=1000, max_value=100000, value=50000)

                pop_size = st.number_input("Tamanho da população", min_value=1, max_value=200, value=50, step=1, key="pop_size")
                epoch = round(maxfes / pop_size)
                submit_button = st.form_submit_button("Executar")

    if not submit_button:
        return

    with col1:
        start_time = time.time()
        agrupados = {
            "dataframes":[],
            "geomeans":[]
        }

        for algoritmo in algoritmos:
            with st.spinner(f"Executando otimizações para {algoritmo}..."):
                for run in range(nruns):

                    f1 = F52017(ndim=30, f_bias=0)
                    f2 = F92017(ndim=30, f_bias=0),
                    f3 = F112017(ndim=30, f_bias=0)
                    
                    run_experiment(algoritmo, p1, pop_size, epoch, False)

                    f1 = F52017(ndim=30, f_bias=0, f_shift='shift_data_2',f_matrix='M_2_D')
                    f2 = F92017(ndim=30, f_bias=0, f_shift='shift_data_2',f_matrix='M_2_D'),
                    f3 = F112017(ndim=30, f_bias=0, f_shift='shift_data_2',f_matrix='M_2_D')

                    run_experiment(algoritmo, p1, pop_size, epoch, True)

            end_time = time.time()
        
        
            # Calcular horas, minutos e segundos que foram necessários para a execução
            tempo_execucao = end_time - start_time
            horas = int(tempo_execucao // 3600)
            minutos = int((tempo_execucao % 3600) // 60)
            segundos = int(tempo_execucao % 60)
            
            # Atualizar o ratio para cada função
            for idx in range(len(resultados["Problema"])):
                resultados["shifted"][idx] = np.mean(resultados["shifted"][idx])
                resultados["unshifted"][idx] = np.mean(resultados["unshifted"][idx])
                resultados["ratio"][idx] = resultados["unshifted"][idx] / resultados["shifted"][idx]
            
            with col1:
                # Criar um dataframe com os resultados
                st.write(f"Resultados para o algoritmo {algoritmo}")
                
                st.write("Média geométrica da razão entre as funções com e sem shift")
                geometric_mean = np.prod(resultados["ratio"]) ** (1 / len(resultados["ratio"]))
                
                # Convertentdo os valores do DataFrame para notaçao científica
                df = pd.DataFrame(resultados)
                df["unshifted"] = df["unshifted"].apply(lambda x: f"{x:.2e}")
                df["shifted"] = df["shifted"].apply(lambda x: f"{x:.2e}")
                df["ratio"] = df["ratio"].apply(lambda x: f"{x:.2e}")
                
                # Adicionando coluna com o nome do algoritmo no dataframe na segunda coluna
                df.insert(0, "Algoritmo", algoritmo)
                
                st.write(df)

                agrupados["dataframes"].append(df)
                

                # Exibindo a média gemoétrica com notaçao científica
                st.write(f"Média Geométrica (ratio): {geometric_mean:.2e}")
                agrupados["geomeans"].append([algoritmo,geometric_mean])

                st.success(f"Execução finalizada em {horas} horas, {minutos} minutos e {segundos} segundos.")
            
    with col2:
        df_unificado = pd.concat(agrupados["dataframes"])
        
        # Criando esquema de cores personalizadas para cada algoritmo
        colors = {"BBO": "dodgerblue", "EP": "goldenrod"}

        # Exibindo gráfico de barras estacadas com os resultados para cada algoritmo X função
        fig = px.bar(df_unificado, x="Problema", y="ratio", title=f"Razão entre as funções com e sem shift para os algoritmos BBO e EP", text="ratio", color="Algoritmo", labels={"ratio": "Razão", "Problema": "Função"}, barmode="group", color_discrete_map=colors)
        fig.update_yaxes(tickformat=".2e")
        st.plotly_chart(fig)

        # Criando um DataFrame com a média geométrica agrupara de cada algoritmo
        medias = pd.DataFrame(agrupados["geomeans"], columns=["Algoritmo", "geomeans"])

        # Exibindo o gráfico de média geométrica para cada algoritmo
        fig = px.bar(medias, x="Algoritmo", y="geomeans", title="Média Geométrica das razões entre as funções com e sem shift para os algoritmos BBO e EP", color="Algoritmo",text="geomeans", labels={"Algoritmo": "Algoritmo", "geomeans": "Média Geométrica"}, color_discrete_map=colors)
        fig.update_yaxes(tickformat=".2e")
        # Atualizando o texto das barras para notação científica
        fig.update_traces(texttemplate="%{text:.2e}")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
