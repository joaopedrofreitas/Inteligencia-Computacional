import utils as utl
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Configurações do experimento
PATHS = [
    '../dataset/ENB2012_data.xlsx',
    '../dataset/ai4i2020.csv'
]
MAX_ROWS = 20       # None para usar todos os dados
SEED = 42
REPETITIONS = 30

if __name__ == "__main__":
    # 1. Leitura de dados
    dfs = utl.read_datasets(PATHS, MAX_ROWS)
    
    # 2. Definir modelos
    models = {
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor()
    }
    
    all_results = {}
    
    # 3. Executar experimentos para cada dataset
    for dataset_name, df in dfs.items():
        if df is None:
            continue
            
        # Pré-processamento
        X, y = utl.preprocess(df, dataset_name)
        
        # Treinar e avaliar cada modelo
        for model_name, model in models.items():
            key = f"{dataset_name}_{model_name}"
            print(f"Executando: {key}")
            
            result = utl.run_regression(
                key,
                model,
                X,
                y,
                REPETITIONS,
                SEED
            )
            all_results[key] = result
            
            # 4. Imprimir resultados
            print(f"\nResultados para {key}:")
            print(f"RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
            print(f"MAE: {result['mae_mean']:.4f} ± {result['mae_std']:.4f}")
            print(f"R²: {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
            print("="*50)
    
    # 5. Gerar visualizações
    utl.plot_results(all_results)
    print("Visualizações salvas no diretório './figures'")
