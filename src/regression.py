import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import os
from sklearn.model_selection import GridSearchCV

def read_datasets(paths, max_rows=None):
    dfs = {}
    
    # Energy Efficiency Dataset
    try:
        df_energy = pd.read_excel(paths[0])
        if max_rows and max_rows < len(df_energy):
            df_energy = df_energy.sample(max_rows, random_state=42)
        dfs['energy'] = df_energy
    except Exception as e:
        print(f"Erro ao ler Energy: {e}")
        dfs['energy'] = None

    # AI4I Predictive Maintenance Dataset
    try:
        df_ai4i = pd.read_csv(paths[1])
        if max_rows and max_rows < len(df_ai4i):
            df_ai4i = df_ai4i.sample(max_rows, random_state=42)
        dfs['ai4i'] = df_ai4i
    except Exception as e:
        print(f"Erro ao ler AI4I: {e}")
        dfs['ai4i'] = None
        
    return dfs

def preprocess(df, dataset_name):
    if dataset_name == 'energy':
        # Energy Efficiency: features e target
        X = df.iloc[:, :8]
        y = df.iloc[:, 8]  # Heating Load (Y1)
        return X, y
    
    elif dataset_name == 'ai4i':
        # AI4I: features e target
        X = df.drop(['UDI', 'Product ID', 'Machine failure', 'HDF', 'Process temperature [K]'], axis=1, errors='ignore')
        y = df['Process temperature [K]']  # Agora usando Process temperature como target
        # One-hot encoding para variáveis categóricas
        X = pd.get_dummies(X, columns=['Type'])
        return X, y
    
    else:
        raise ValueError(f"Dataset de regressão desconhecido: {dataset_name}")

def run_regression(name, model, X, y, repetitions, seed):
    rmse_list = []
    mae_list = []
    r2_list = []

    # Otimização de hiperparâmetros para SVR, RandomForest e GradientBoosting
    best_model = model
    model_name = model.__class__.__name__
    if model_name == 'SVR':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_scaled, y)
        best_model = grid.best_estimator_
        print(f"Melhores hiperparâmetros para SVR em {name}: {grid.best_params_}")
    elif model_name == 'RandomForestRegressor':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_scaled, y)
        best_model = grid.best_estimator_
        print(f"Melhores hiperparâmetros para RandomForest em {name}: {grid.best_params_}")
    elif model_name == 'GradientBoostingRegressor':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        }
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_scaled, y)
        best_model = grid.best_estimator_
        print(f"Melhores hiperparâmetros para GradientBoosting em {name}: {grid.best_params_}")

    for rep in range(repetitions):
        # Split com seed incremental
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed + rep
        )
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Clone do modelo para estado limpo
        model_clone = clone(best_model)
        
        # Definir random state se disponível
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = seed + rep
        
        # Treinar e predizer
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)
        
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
        'rmse_list': rmse_list,
        'mae_list': mae_list,
        'r2_list': r2_list,
        'model': best_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def plot_results(results, save_dir='./figures'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot de comparação de modelos
    plt.figure(figsize=(12, 6))
    models = [key.split('_')[-1] for key in results.keys()]
    rmse_means = [res['rmse_mean'] for res in results.values()]
    
    sns.barplot(x=models, y=rmse_means)
    plt.title('Comparação RMSE Entre Modelos')
    plt.ylabel('RMSE Médio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rmse_comparison.png')
    plt.close()
    
    # Plots individuais dos modelos
    for key, result in results.items():
        base_name, model_name = key.split('_')
        
        # Predito vs Real
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
        
        # Distribuição de erro
        errors = result['y_test'] - result['y_pred']
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f'{model_name} - Distribuição de Erro ({base_name})')
        plt.xlabel('Erro de Predição')
        plt.savefig(f'{save_dir}/{key}_error_dist.png')
        plt.close() 