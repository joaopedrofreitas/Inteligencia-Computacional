import classification as classification
import regression as regression
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from ml_utils import create_mlp
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

REGRESSION_PATHS = ['../dataset/ENB2012_data.xlsx', '../dataset/ai4i2020.csv']
CLASSIFICATION_PATHS = ['../dataset/students.csv', '../dataset/winequality-red.csv']
MAX_ROWS = None
SEED = 42
REPETITIONS = 30

# Substituir LinearRegression por KernelRidge
def run_regression_experiment():
    print("="*60)
    print("EXPERIMENTO DE REGRESSÃO")
    print("="*60)
    
    dfs = regression.read_datasets(REGRESSION_PATHS)
    models = {
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=SEED),
        'GradientBoosting': GradientBoostingRegressor(random_state=SEED),
        'MLPRegressor': 'keras_mlp'  # Placeholder para tratamento especial
    }
    results = {}
    
    for dataset_name, df in dfs.items():
        if df is None:
            continue
        
        X, y = regression.preprocess(df, dataset_name)
        print(f"Escala do valor alvo ({dataset_name}): min={y.min():.2f}, max={y.max():.2f}, média={y.mean():.2f}")
        n_inputs = X.shape[1]
        input_shape = (n_inputs,)
        
        for model_name, model in models.items():
            key = f"{dataset_name}_{model_name}"
            print(f"Executando: {key}")
            
            if model_name == 'MLPRegressor':
                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                X = np.nan_to_num(X)
                y = np.nan_to_num(y)
                if isinstance(X, pd.DataFrame):
                    if len(X.select_dtypes(include=['object', 'category']).columns) > 0:
                        X = pd.get_dummies(X)
                    X = X.values.astype(np.float32)
                rmse_list = []
                mae_list = []
                r2_list = []
                rmse_norm_list = []
                mae_norm_list = []
                for rep in range(REPETITIONS):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=SEED + rep
                    )
                    model, norm = create_mlp(input_shape)
                    norm.adapt(X_train)
                    history = model.fit(
                        X_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=32,
                        verbose=0
                    )
                    y_pred = model.predict(X_test).flatten()
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    y_range = np.max(y_test) - np.min(y_test)
                    if y_range == 0:
                        rmse_norm = 0
                        mae_norm = 0
                    else:
                        rmse_norm = rmse / y_range
                        mae_norm = mae / y_range
                    rmse_list.append(rmse)
                    mae_list.append(mae)
                    r2_list.append(r2)
                    rmse_norm_list.append(rmse_norm)
                    mae_norm_list.append(mae_norm)
                result = {
                    'rmse_mean': np.mean(rmse_list),
                    'rmse_std': np.std(rmse_list),
                    'mae_mean': np.mean(mae_list),
                    'mae_std': np.std(mae_list),
                    'r2_mean': np.mean(r2_list),
                    'r2_std': np.std(r2_list),
                    'rmse_norm_mean': np.mean(rmse_norm_list),
                    'rmse_norm_std': np.std(rmse_norm_list),
                    'mae_norm_mean': np.mean(mae_norm_list),
                    'mae_norm_std': np.std(mae_norm_list),
                    'rmse_list': rmse_list,
                    'mae_list': mae_list,
                    'r2_list': r2_list,
                    'rmse_norm_list': rmse_norm_list,
                    'mae_norm_list': mae_norm_list,
                    'model': model,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
            else:
                result = regression.run_regression(key, model, X, y, REPETITIONS, SEED)
            results[key] = result
            
            print(f"\nResultados para {key}:")
            print(f"RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
            print(f"MAE: {result['mae_mean']:.4f} ± {result['mae_std']:.4f}")
            print(f"R²: {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
            print("-"*50)
    
    regression.plot_results(results)
    print("Gráficos salvos em './figures'")

def run_classification_experiment():
    print("="*60)
    print("EXPERIMENTO DE CLASSIFICAÇÃO MULTICLASSE")
    print("="*60)
    
    dfs = classification.read_classification_datasets(CLASSIFICATION_PATHS)
    models = {
        'RandomForest': RandomForestClassifier(random_state=SEED),
        'XGBoost': XGBClassifier(random_state=SEED),
        'GradientBoosting': GradientBoostingClassifier(random_state=SEED),
        'FuzzyKNN': classification.OptimizedFuzzyKNNClassifier(
            k_range=(3, 15), m=2.0, auto_k=True, 
            adaptive_weighting=True, batch_size=500
        )
    }
    results = {}
    
    for dataset_name, df in dfs.items():
        if df is None:
            continue
            
        print(f"\n{'='*60}")
        print(f"PROCESSANDO DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        # Definir threshold de outlier
        if dataset_name.lower().startswith('wine'):
            X, y = classification.preprocess_classification(df, dataset_name)
            outlier_threshold = 3
        else:
            X, y = classification.preprocess_classification(df, dataset_name)
            outlier_threshold = 1.5
        
        n_classes = len(np.unique(y))
        max_k = min(15, len(X) // 4)
        models['FuzzyKNN'] = classification.OptimizedFuzzyKNNClassifier(
            k_range=(3, max_k), m=2.0, auto_k=True, 
            adaptive_weighting=False, batch_size=500
        )
        
        print(f"Número de classes: {n_classes}")
        print(f"Classes: {sorted(np.unique(y))}")
        
        dataset_results = {}
        
        for model_name, model in models.items():
            key = f"{dataset_name}_{model_name}"
            print(f"\nExecutando: {key}")
            
            result = classification.run_classification(key, model, X, y, REPETITIONS, SEED, use_advanced_preprocessing=True, outlier_threshold=outlier_threshold)
            dataset_results[key] = result
            results[key] = result
            
            print(f"\nResultados para {key}:")
            print(f"Acurácia: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
            print(f"F1-Score: {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
            print("-"*50)
        
        print(f"\nANÁLISE ESTATÍSTICA - DATASET: {dataset_name}")
        print("="*50)
        classification.statistical_analysis(dataset_results, dataset_name=dataset_name)
        classification.plot_classification_results(dataset_results, dataset_name=dataset_name)
    
    print("Gráficos salvos em './figures'") 