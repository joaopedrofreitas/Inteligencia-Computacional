# Projeto de Inteligência Computacional

Este projeto implementa experimentos de machine learning para problemas de regressão e classificação multiclasse, comparando diferentes algoritmos e técnicas de pré-processamento.

## Datasets

O projeto trabalha com os seguintes datasets:

### Regressão
- **Energy Efficiency Dataset**: Predição de carga de aquecimento baseada em características de construção
- **AI4I Predictive Maintenance**: Predição de temperatura do ar baseada em sensores de máquina

### Classificação
- **Students Dataset**: Classificação de desempenho acadêmico
- **Wine Quality**: Classificação da qualidade do vinho

## Algoritmos Implementados

### Regressão
- Regressão Linear
- Support Vector Regression (SVR)
- Random Forest
- Gradient Boosting

### Classificação
- Random Forest
- XGBoost
- Gradient Boosting
- Fuzzy K-NN (implementação customizada)

## Funcionalidades

- **Pré-processamento avançado**: Remoção de outliers, balanceamento de classes, normalização robusta
- **Otimização de hiperparâmetros**: Grid search para modelos suportados
- **Análise estatística**: Testes de Friedman e Nemenyi para comparação de modelos
- **Visualizações**: Gráficos de comparação, matrizes de confusão, importância de features

## Como Usar

Use a versão 3.9 do python

1. Instale as dependências:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn scipy
```

2. Execute o experimento:
```bash
cd src
py -3.9 main.py

```

## Windows
```bash
winget install python.python.3.9
py -3.9 -m pip install "tensorflow<2.11"
# Instale as otras dependências e use numpy==1.24.4
cd src
py -3.9 main.py
```
3. Escolha o tipo de experimento:
   - 1: Regressão
   - 2: Classificação Multiclasse

## Estrutura do Projeto

```
src/
├── main.py              # Script principal
├── utils.py             # Utilitários para classificação
├── regression_utils.py  # Utilitários para regressão
└── figures/             # Gráficos gerados
```

## Resultados

Os resultados são salvos na pasta `figures/` e incluem:
- Comparações de performance entre modelos
- Matrizes de confusão
- Análises de importância de features
- Distribuições de erro (regressão)

## Implementação Fuzzy K-NN

O algoritmo Fuzzy K-NN implementado inclui:
- Seleção automática de K usando método do cotovelo
- Pesos adaptativos baseados em importância de features
- Processamento em lotes para eficiência
- Compatibilidade com scikit-learn
