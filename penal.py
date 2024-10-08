import time
import numpy as np
from mealpy.bio_based import BBO
from mealpy.evolutionary_based import EP
from mealpy import FloatVar
import plotly.express as px
import pandas as pd
import streamlit as st

class OptimizerWithPenalty:
    def __init__(self, constraint_count, optimization_algorithm=EP.OriginalEP, execution_count=30, evaluation_count=10000):
        """
        Construtor.
        Parameters:
        - constraint_count: número de restrições do problema.
        - optimization_algorithm: classe do algoritmo de otimização (ex: EP ou BBO da biblioteca Mealpy)
        - execution_count: número de execuções independentes para a otimização.
        - evaluation_count: número total de avaliações da função objetivo.
        """
        self.constraint_count = constraint_count
        self.optimization_algorithm = optimization_algorithm
        self.execution_count = execution_count
        self.evaluation_count = evaluation_count

    def objective_function(self, variables):
        # Definindo o problema
        x1, x2, x3, x4 = variables

        def constraint1(x):
            return x1 - 0.0193*x3
        def constraint2(x):
            return x2 - 0.00954*x3
        def constraint3(x):
            return 3.14*(x3**2)*x4 + ((4*3.14*(x3**3))/3) - 1296000
        def constraint4(x):
            return -x4 + 240

        violations = [constraint1(variables), constraint2(variables), constraint3(variables), constraint4(variables)]
        violations = [violation if violation > 0 else 0 for violation in violations]

        V = (0.6224*x1*x2*x3) + (1.7781*x2*(x3**2)) + (3.1661*((x1)**2)*x3) 
        return V, violations  # Certifique-se de que o valor de V é o primeiro item retornado

    def custom_penalty(self, solution, objective_func, constraints, current_population):
        """
        Função de penalização personalizada com base na lógica de Kalyanmoy Deb.
        """
        # Verificar se a solução é viável (todas as restrições são satisfeitas)
        feasible = all([constraint(solution) >= 0 for constraint in constraints])
       
        if feasible:
            return objective_func(solution)  # Retorna o valor da função objetiva se a solução for viável
        else:
            # Obtendo fmax, que é o pior valor objetivo das soluções viáveis
            feasible_solutions = [sol for sol in current_population if all(constraint(sol) >= 0 for constraint in constraints)]
            if feasible_solutions:
                f_max = max([objective_func(sol) for sol in feasible_solutions])
            else:
                f_max = 100000  # Valor alto padrão se nenhuma solução viável estiver presente
            
            # Soma das penalidades das violações das restrições
            penalty = sum([abs(constraint(solution)) if constraint(solution) < 0 else 0 for constraint in constraints])
            return f_max + penalty

    def penalized_fitness(self, solution, population):
        """
        Função objetiva penalizada que calcula o fitness usando a penalização personalizada.
        """
        # Definir restrições
        constraints = [
            lambda x: x[0] - 0.0193*x[2],  # g1
            lambda x: x[1] - 0.00954*x[2],  # g2
            lambda x: 3.14*(x[2]**2)*x[3] + ((4*3.14*(x[2]**3))/3) - 1296000,  # g3
            lambda x: -x[3] + 240  # g4
        ]

        # Aplicar a função de penalização personalizada
        fitness_value = self.custom_penalty(solution, lambda x: self.objective_function(x)[0], constraints, population)

        return fitness_value  # Retorna o valor do fitness (um escalar)

    def execute_optimization(self, lower_bounds, upper_bounds, population_size=50):
        """
        Executa a otimização múltiplas vezes e retorna as métricas para o V.
        """
        # Calcular o número de epochs com base no número total de avaliações e no tamanho da população
        epochs = self.evaluation_count // population_size

        results = []
        for _ in range(self.execution_count):
            # Inicializar a população
            population = np.random.uniform(lower_bounds, upper_bounds, (population_size, len(lower_bounds)))

            # Otimização usando o algoritmo passado com a função objetiva penalizada
            problem = {
                "obj_func": lambda solution: self.penalized_fitness(solution, population),
                "bounds": FloatVar(lb=lower_bounds, ub=upper_bounds),
                "minmax": "min",
                "log_to": None,
            }

            model = self.optimization_algorithm(epoch=epochs, pop_size=population_size)
            model.solve(problem)

            best_solution = model.g_best.solution
            best_fitness = model.g_best.target.fitness

            # Armazenar os valores de V para a melhor solução
            V_best, _ = self.objective_function(best_solution)
            results.append(V_best)

        # Calcular as métricas
        best_value = np.min(results)
        median_value = np.median(results)
        mean_value = np.mean(results)
        std_dev = np.std(results)
        worst_value = np.max(results)

        return best_value, median_value, mean_value, std_dev, worst_value

def main():

    st.set_page_config(page_title="Otimização de EP e BBO com Penalização Personalizada", page_icon="📊", layout="wide")

    # Parâmetros do problema
    constraint_count = 3

    optimization_algorithms = {"EP": EP.OriginalEP, "BBO": BBO.OriginalBBO}

    # Interface gráfica
    st.title("Otimização de EP e BBO com Penalização Personalizada")
    st.sidebar.title("Configurações")

    with st.sidebar:
        with st.form(key="config_form"):
            execution_count = st.number_input("Número de execuções", min_value=1, max_value=100, value=35, step=1, key="execution_count")
            evaluation_count = st.number_input("Número total de avaliações", min_value=1000, max_value=100000, value=36000, step=1000, key="evaluation_count")
            population_size = st.number_input("Tamanho da população", min_value=1, max_value=200, value=50, step=1, key="population_size")
            submit_button = st.form_submit_button("Executar")

    if not submit_button:
        return
    
    # Limites das variáveis de decisão
    lower_bounds = [0.00625, 0.00625, 10, 10]
    upper_bounds = [5, 5, 200, 200]

    results = []
    col1, col2 = st.columns(2)

    # Percorrer cada algoritmo de otimização
    with st.spinner("Executando otimizações..."):
        start_time = time.time()
        for key in optimization_algorithms.keys():
            optimizer = OptimizerWithPenalty(
                constraint_count=constraint_count,
                optimization_algorithm=optimization_algorithms[key],
                execution_count=execution_count,
                evaluation_count=evaluation_count
            )
            # Executar a otimização e obter as métricas
            best_value, median_value, mean_value, std_dev, worst_value = optimizer.execute_optimization(lower_bounds, upper_bounds, population_size=population_size)
            results.append((key, best_value, median_value, mean_value, std_dev, worst_value))
        end_time = time.time()
        # Calcular horas, minutos e segundos que foram necessários para a execução
        execution_time = end_time - start_time
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)

    st.success(f"Execução finalizada em {hours} horas, {minutes} minutos e {seconds} segundos.")  

    # Criar dataframe com os resultados
    df = pd.DataFrame(results, columns=["Algoritmo", "Melhor", "Mediana", "Média", "Desvio Padrão", "Pior"])

    col1.write("Resultados")
    col1.write(df)
    # Gráfico de barras
    fig = px.bar(df, x="Algoritmo", y="Melhor", color="Algoritmo", title="Melhor valor de V para cada algoritmo")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
