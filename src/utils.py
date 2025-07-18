import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import os

# Leitura de dados
def read_datasets(paths: list[str], max_rows: int = None) -> dict[str, pd.DataFrame]:
    """
    Lê os datasets e retorna um dicionário com os DataFrames
    Chaves: 'energy', 'ai4i'
    """
    dfs = {}
    
    # Energy Efficiency Dataset
    try:
        df_energy = pd.read_excel(paths[0])
        if max_rows and max_rows < len(df_energy):
            df_energy = df_energy.sample(max_rows, random_state=42)
        dfs['energy'] = df_energy
    except Exception as e:
        print(f"Erro ao ler Energy dataset: {e}")
        dfs['energy'] = None

    # AI4I Predictive Maintenance Dataset
    try:
        df_ai4i = pd.read_csv(paths[1])
        if max_rows and max_rows < len(df_ai4i):
            df_ai4i = df_ai4i.sample(max_rows, random_state=42)
        dfs['ai4i'] = df_ai4i
    except Exception as e:
        print(f"Erro ao ler AI4I dataset: {e}")
        dfs['ai4i'] = None
        
    return dfs

# Pré-processamento
def preprocess(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pré-processamento específico para cada dataset
    Retorna (X, y)
    """
    if dataset_name == 'energy':
        # Energy Efficiency: selecionar features e target
        X = df.iloc[:, :8]
        y = df.iloc[:, 8]  # Heating Load (Y1)
        return X, y
    
    elif dataset_name == 'ai4i':
        # AI4I: selecionar features e target
        X = df.drop(['UDI', 'Product ID', 'Machine failure'], axis=1, errors='ignore')
        y = df['Air temperature [K]']  # Exemplo de variável contínua
        # One-hot encoding para variáveis categóricas
        X = pd.get_dummies(X, columns=['Type'])
        return X, y
    
    else:
        raise ValueError(f"Dataset desconhecido: {dataset_name}")

# Execução de regressão
def run_regression(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    repetitions: int,
    seed: int
) -> dict[str, float]:
    """
    Executa repetições do experimento de regressão
    Retorna dicionário com métricas
    """
    rmse_list = []
    mae_list = []
    r2_list = []
    
    for rep in range(repetitions):
        # Split com semente incremental
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed+rep
        )
        
        # Normalização
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Clonar modelo para garantir estado limpo
        model_clone = clone(model)
        
        # Configurar random_state se existir
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = seed + rep
            
        # Treinamento e predição
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        # Calcular métricas
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        r2_list.append(r2_score(y_test, y_pred))
    
    return {
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'r2_mean': np.mean(r2_list),
        'r2_std': np.std(r2_list),
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

# Visualização de resultados
def plot_results(results: dict, save_dir: str = './figures'):
    """
    Gera e salva gráficos de análise
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Gráfico de comparação de modelos
    plt.figure(figsize=(12, 6))
    models = [key.split('_')[-1] for key in results.keys()]
    rmse_means = [res['rmse_mean'] for res in results.values()]
    
    sns.barplot(x=models, y=rmse_means)
    plt.title('Comparação de RMSE entre Modelos')
    plt.ylabel('RMSE Médio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rmse_comparison.png')
    plt.close()
    
    # Gráficos individuais por modelo
    for key, result in results.items():
        base_name, model_name = key.split('_')
        
        # Predição vs Real
        plt.figure(figsize=(10, 6))
        plt.scatter(result['y_test'], result['y_pred'], alpha=0.5)
        plt.plot([min(result['y_test']), max(result['y_test'])], 
                 [min(result['y_test']), max(result['y_test'])], 'r--')
        plt.title(f'{model_name} - Predito vs Real ({base_name})')
        plt.xlabel('Valor Real')
        plt.ylabel('Predição')
        plt.grid(True)
        plt.savefig(f'{save_dir}/{key}_pred_vs_real.png')
        plt.close()
        
        # Distribuição de Erros
        errors = result['y_test'] - result['y_pred']
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f'{model_name} - Distribuição de Erros ({base_name})')
        plt.xlabel('Erro de Predição')
        plt.savefig(f'{save_dir}/{key}_error_dist.png')
        plt.close()
