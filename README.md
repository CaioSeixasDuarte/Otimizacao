## Pré-requisitos

Para reproduzir os experimentos, você precisará ter o Python 3.8+ instalado em seu ambiente. Recomenda-se utilizar um ambiente virtual para gerenciar as dependências.

## Instalação

1. Clone este repositório em sua máquina local:

2. Crie um ambiente virtual e ative-o:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/MacOS
    venv\Scripts\activate  # Para Windows
    ```

3. Instale as dependências listadas no `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Executando os Experimentos

### Experimento #1

O script `exp_1.py` permite reproduzir o primeiro experimento, que analisa a eficácia dos algoritmos EP e BBO em problemas de otimização com viés central e viés zero.

Para executar o experimento:

```bash
streamlit run exp_1.py
```

### Experimento #2

O script `exp_2.py` permite reproduzir o segundo experimento, que aplica estratégias de penalização adaptativa ao problema de vaso de pressão.

Para executar o experimento:

```bash
streamlit run exp_2.py
```

