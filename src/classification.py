import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist   
from scipy.stats import friedmanchisquare


def read_classification_datasets(paths, max_rows=None):
    dfs = {}
    
    # Students Dataset
    try:
        df_students = pd.read_csv(paths[0])
        if max_rows and max_rows < len(df_students):
            df_students = df_students.sample(max_rows, random_state=42)
        dfs['students'] = df_students
    except Exception as e:
        print(f"Erro ao ler Students: {e}")
        dfs['students'] = None

    # Wine Quality Dataset
    try:
        df_wine = pd.read_csv(paths[1])
        if max_rows and max_rows < len(df_wine):
            df_wine = df_wine.sample(max_rows, random_state=42)
        dfs['wine'] = df_wine
    except Exception as e:
        print(f"Erro ao ler Wine: {e}")
        dfs['wine'] = None
        
    return dfs

def preprocess_classification(df, dataset_name):
    if dataset_name == 'students':
        # Students: features e target
        if 'Target' in df.columns:
            y = df['Target']
            X = df.drop(['Target'], axis=1, errors='ignore')
        else:
            # Usar última coluna como target se não houver coluna 'Target'
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
        
        # One-hot encoding para variáveis categóricas
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            X = pd.get_dummies(X, columns=categorical_columns)
        
        # Label encoding para variável target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y = pd.Series(y_encoded, index=y.index, name=y.name)
        
        return X, y
    
    elif dataset_name == 'wine':
        # Wine Quality: 'quality' é a variável target
        y = df['quality']
        X = df.drop(['quality'], axis=1, errors='ignore')
        
        # Re-codificar classes para começar do 0 para compatibilidade sklearn
        unique_classes = sorted(y.unique())
        class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
        y = y.map(class_mapping)
        
        # Definir threshold de outlier como 2.5 para wine
        X.outlier_threshold = 2.5
        
        return X, y
    else:
        raise ValueError(f"Dataset de classificação desconhecido: {dataset_name}")

def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """Remove outliers using IQR method"""
    print("Removing outliers...")
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
            print(f"Removendo {outliers.sum()} outliers de {col}")
            
            df_clean = df_clean[~outliers]
    
    print(f"Shape do dataset após remoção de outliers: {df_clean.shape}")
    return df_clean

def balance_classes(X, y, method='smote'):
    print("Balanceamento...")
    
    try:
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.combine import SMOTETomek, SMOTEENN
        
        print(f"Distribuição original: {np.bincount(y)}")
        
        # Verificar se alguma classe tem poucos exemplos para SMOTE
        class_counts = np.bincount(y)
        min_samples = min(class_counts[class_counts > 0])
        
        if min_samples < 6:
            print(f"Aviso: Algumas classes têm apenas {min_samples} exemplos. Pulando SMOTE.")
            return X, y
        
        # Escolher método de balanceamento baseado nas características do dataset
        if method == 'smote':
            balancer = SMOTE(random_state=42)
        elif method == 'adasyn':
            balancer = ADASYN(random_state=42)
        elif method == 'borderline_smote':
            balancer = BorderlineSMOTE(random_state=42)
        elif method == 'smote_tomek':
            balancer = SMOTETomek(random_state=42)
        elif method == 'smote_enn':
            balancer = SMOTEENN(random_state=42)
        else:
            balancer = SMOTE(random_state=42)
        
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
        print(f"Distribuição balanceada: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    except ImportError:
        print("Aviso: imbalanced-learn não disponível.")
        return X, y

def optimize_hyperparameters(X, y, model_name, model, random_state=42):
    print(f"Otimizando {model_name}...")
    
    # Grids de parâmetros reduzidos para diferentes modelos
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt','log2']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.1, 1.0],
            'reg_lambda': [0.1, 1.0]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
    
    if model_name in param_grids:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scoring = 'f1_weighted'
        
        grid_search = GridSearchCV(
            model, param_grids[model_name], cv=cv, scoring=scoring,
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        print(f"Score CV: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        return model

class OptimizedFuzzyKNNClassifier:
    
    def __init__(self, k_range=(3, 20), m=2.0, distance_metric='euclidean', 
                 auto_k=True, outlier_threshold=0.1, adaptive_weighting=True,
                 use_cache=True, batch_size=1000):
        self.k_range = k_range
        self.m = m
        self.distance_metric = distance_metric
        self.auto_k = auto_k
        self.outlier_threshold = outlier_threshold
        self.adaptive_weighting = adaptive_weighting
        self.use_cache = use_cache
        self.batch_size = batch_size
        
        # Modelo
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.optimal_k = None
        self.feature_weights = None
        self.density_weights = None
        self._distance_cache = {}
        
    def get_params(self, deep=True):
        return {
            'k_range': self.k_range,
            'm': self.m,
            'distance_metric': self.distance_metric,
            'auto_k': self.auto_k,
            'outlier_threshold': self.outlier_threshold,
            'adaptive_weighting': self.adaptive_weighting,
            'use_cache': self.use_cache,
            'batch_size': self.batch_size
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def _elbow_method(self, X, y):
        k_min, k_max = self.k_range
        k_values = range(k_min, min(k_max + 1, len(X) // 2))
        
        # Usar amostra menor
        sample_size = min(1000, len(X))
        if sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        avg_distances = []
        
        for k in k_values:
            nbrs = NearestNeighbors(n_neighbors=k+1, metric=self.distance_metric, algorithm='ball_tree')
            nbrs.fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            
            avg_dist = np.mean(distances[:, 1:])
            avg_distances.append(avg_dist)
        
        # Encontrar cotovelo
        if len(avg_distances) > 2:
            first_diff = np.diff(avg_distances)
            second_diff = np.diff(first_diff)
            
            elbow_idx = np.argmax(np.abs(second_diff)) + 1
            optimal_k = k_values[elbow_idx]
        else:
            optimal_k = k_values[0]
        
        return optimal_k
    
    def _calculate_feature_weights(self, X, y):
        try:
            X_np = np.asarray(X, dtype=np.float64)
            y_np = np.asarray(y, dtype=np.int64)
            
            mi_scores = mutual_info_classif(X_np, y_np, random_state=42)
            
            if np.sum(mi_scores) > 0:
                weights = mi_scores / np.sum(mi_scores)
            else:
                weights = np.ones(X_np.shape[1]) / X_np.shape[1]
            
            return weights
            
        except Exception as e:
            print(f"Aviso: Erro ao calcular pesos das features: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _calculate_density_weights(self, X):
        try:
            # Usar K-means para estimar densidade local
            n_clusters = min(10, len(X) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calcular tamanhos dos clusters como pesos de densidade
            cluster_sizes = np.bincount(cluster_labels)
            density_weights = cluster_sizes[cluster_labels]
            
            # Normalizar pesos
            density_weights = density_weights / np.max(density_weights)
            
            return density_weights
            
        except Exception as e:
            print(f"Aviso: Erro ao calcular pesos de densidade: {e}")
            return np.ones(len(X))
    
    def _weighted_distance_matrix(self, X1, X2):
        if self.feature_weights is not None:
            # Aplicar pesos das features
            X1_weighted = X1 * np.sqrt(self.feature_weights)
            X2_weighted = X2 * np.sqrt(self.feature_weights)
        else:
            X1_weighted, X2_weighted = X1, X2
        
        # Calcular distâncias
        distances = cdist(X1_weighted, X2_weighted, metric=self.distance_metric)
        
        return distances
    
    def _fuzzy_membership(self, distances, k):
        # Calcular graus de pertinência fuzzy
        # Encontrar k vizinhos mais próximos
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
        k_nearest_distances = np.take_along_axis(distances, k_nearest_indices, axis=1)
        
        # Calcular graus de pertinência
        membership = np.zeros((len(distances), len(self.classes_)))
        
        for i in range(len(distances)):
            for j, neighbor_idx in enumerate(k_nearest_indices[i]):
                class_label = self.y_train[neighbor_idx]
                distance = k_nearest_distances[i, j]
                
                if distance == 0:
                    membership[i, class_label] += 1.0
                else:
                    membership[i, class_label] += 1.0 / (distance ** (2 / (self.m - 1)))
        
        row_sums = membership.sum(axis=1, keepdims=True)
        membership = np.divide(membership, row_sums, out=np.zeros_like(membership), where=row_sums != 0)
        
        return membership
    
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(y)
        
        if self.auto_k:
            self.optimal_k = self._elbow_method(self.X_train, self.y_train)
        else:
            self.optimal_k = self.k_range[0]
        
        if self.adaptive_weighting:
            self.feature_weights = self._calculate_feature_weights(self.X_train, self.y_train)
            self.density_weights = self._calculate_density_weights(self.X_train)
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        predictions = []
        
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i + self.batch_size]
            batch_predictions = self._predict_batch(batch)
            predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def _predict_batch(self, X_batch):
        distances = self._weighted_distance_matrix(X_batch, self.X_train)
        
        membership = self._fuzzy_membership(distances, self.optimal_k)
        
        # Predizer classe com maior pertinência
        predictions = np.argmax(membership, axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        X = np.asarray(X)
        
        distances = self._weighted_distance_matrix(X, self.X_train)
        membership = self._fuzzy_membership(distances, self.optimal_k)
        
        return membership

def run_classification(name, model, X, y, repetitions, seed, use_advanced_preprocessing=False, outlier_threshold=1.5):
    if use_advanced_preprocessing:
        print("Pré-processamento avançado...")
        
        # Limpeza e remoção de outliers
        X_clean = X.dropna()
        y_clean = y[X_clean.index]
        
        numeric_columns = X_clean.select_dtypes(include=['float64', 'int64']).columns
        X_clean = remove_outliers(X_clean, numeric_columns, threshold=outlier_threshold)
        y_clean = y_clean[X_clean.index]
        
        # Normalização e balanceamento
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_clean)
        X_balanced, y_balanced = balance_classes(X_scaled, y_clean)
        X_selected = X_balanced
        
        # Otimização de hiperparâmetros
        model_name = name.split('_')[-1] if '_' in name else name
        if model_name in ['RandomForest', 'XGBoost', 'GradientBoosting']:
            optimized_model = optimize_hyperparameters(X_selected, y_balanced, model_name, model, seed)
        else:
            optimized_model = clone(model)
    else:
        X_selected = X
        y_balanced = y
        optimized_model = clone(model)
    
    accuracy_list = []
    f1_list = []
    
    for rep in range(repetitions):
        # Split com seed incremental
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_balanced, test_size=0.3, random_state=seed + rep, stratify=y_balanced
        )
        
        if not use_advanced_preprocessing:
            # Aplicar pré-processamento padrão para cada repetição
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Clone do modelo para esta repetição
        model_clone = clone(optimized_model)
        
        # Definir random state se disponível
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = seed + rep
            
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        accuracy_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred, average='weighted'))
    
    return {
        'accuracy_mean': np.mean(accuracy_list),
        'accuracy_std': np.std(accuracy_list),
        'f1_mean': np.mean(f1_list),
        'f1_std': np.std(f1_list),
        'accuracy_list': accuracy_list,
        'f1_list': f1_list,
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def nemenyi_test(data_groups, model_names):
    # Teste post-hoc de Nemenyi para comparar todos os pares de modelos
    
    # Converter dados para ranks
    all_data = np.concatenate(data_groups)
    ranks = rankdata(all_data)
    
    # Dividir ranks de volta para grupos
    start_idx = 0
    rank_groups = []
    for group in data_groups:
        end_idx = start_idx + len(group)
        rank_groups.append(ranks[start_idx:end_idx])
        start_idx = end_idx
    
    # Calcular ranks médios
    mean_ranks = [np.mean(ranks) for ranks in rank_groups]
    
    # Diferença crítica de Nemenyi
    k = len(data_groups)  # número de modelos
    n = len(data_groups[0])  # número de repetições
    q_alpha = 2.569  # Valor crítico para alpha=0.05 (aproximado)
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * n))
    
    # Comparar todos os pares
    significant_pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            if rank_diff > cd:
                significant_pairs.append((model_names[i], model_names[j], rank_diff))
    
    return significant_pairs, mean_ranks, cd

def statistical_analysis(results, dataset_name=None):
    dataset_prefix = f"[{dataset_name}] " if dataset_name else ""
    print(f"\n{dataset_prefix}Análise Estatística:")
    print("="*50)
    
    # Extrair scores de acurácia e F1
    datasets = set()
    models = set()
    
    for key in results.keys():
        dataset, model = key.split('_', 1)
        datasets.add(dataset)
        models.add(model)
    
    # Comparar modelos entre datasets
    for metric in ['accuracy', 'f1']:
        print(f"\n{dataset_prefix}Comparação {metric.upper()}:")
        print("-" * 30)
        
        for dataset in sorted(datasets):
            print(f"\n{dataset.upper()} Dataset:")
            dataset_results = []
            
            for model in sorted(models):
                key = f"{dataset}_{model}"
                if key in results:
                    mean_val = results[key][f'{metric}_mean']
                    std_val = results[key][f'{metric}_std']
                    dataset_results.append((model, mean_val, std_val))
                    print(f"  {model}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Teste estatístico
            if len(dataset_results) > 1:
                repetition_data = []
                model_names = []
                for model, _, _ in dataset_results:
                    key = f"{dataset}_{model}"
                    if key in results and f'{metric}_list' in results[key]:
                        repetition_data.append(results[key][f'{metric}_list'])
                        model_names.append(model)
                
                if len(repetition_data) > 1 and all(len(data) == len(repetition_data[0]) for data in repetition_data):
                    try:
                        # Teste de Friedman para múltiplas comparações
                        statistic, p_value = friedmanchisquare(*repetition_data)
                        print(f"  Teste de Friedman p-value: {p_value:.4f}")
                        
                        if p_value < 0.05:
                            print(f"  Diferenças significativas entre modelos (p < 0.05)")
                            
                            # Teste post-hoc de Nemenyi
                            significant_pairs, mean_ranks, cd = nemenyi_test(repetition_data, model_names)
                            
                            print(f"  Teste post-hoc de Nemenyi (diferença crítica = {cd:.3f}):")
                            if significant_pairs:
                                for model1, model2, rank_diff in significant_pairs:
                                    print(f"    {model1} vs {model2}: significativo (diff rank = {rank_diff:.3f})")
                            else:
                                print(f"    Nenhuma diferença significativa encontrada")
                                
                            # Mostrar ranks médios
                            print(f"  Ranks médios:")
                            for model, rank in zip(model_names, mean_ranks):
                                print(f"    {model}: {rank:.3f}")
                        else:
                            print(f"  Nenhuma diferença significativa entre modelos (p ≥ 0.05)")
                    except Exception as e:
                        print(f"  Erro no teste estatístico: {e}")
                else:
                    print(f"  Dados insuficientes para teste estatístico")

def plot_classification_results(results, dataset_name=None, save_dir='./figures'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    

    
    # Extract data for plotting
    datasets = set()
    models = set()
    
    for key in results.keys():
        dataset, model = key.split('_', 1)
        datasets.add(dataset)
        models.add(model)
    
    # Create dataset-specific filename prefix
    filename_prefix = f"{dataset_name}_" if dataset_name else ""
    
    # Accuracy comparison plot
    plt.figure(figsize=(12, 6))
    accuracy_data = []
    model_names = []
    
    for model in sorted(models):
        for dataset in sorted(datasets):
            key = f"{dataset}_{model}"
            if key in results:
                accuracy_data.append(results[key]['accuracy_mean'])
                model_names.append(f"{dataset}_{model}")
    
    sns.barplot(x=model_names, y=accuracy_data)
    title = f'Classification Accuracy Comparison - {dataset_name}' if dataset_name else 'Classification Accuracy Comparison'
    plt.title(title)
    plt.ylabel('Mean Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename_prefix}classification_accuracy_comparison.png')
    plt.close()
    
    # F1-score comparison plot
    plt.figure(figsize=(12, 6))
    f1_data = []
    
    for model in sorted(models):
        for dataset in sorted(datasets):
            key = f"{dataset}_{model}"
            if key in results:
                f1_data.append(results[key]['f1_mean'])
    
    sns.barplot(x=model_names, y=f1_data)
    title = f'Classification F1-Score Comparison - {dataset_name}' if dataset_name else 'Classification F1-Score Comparison'
    plt.title(title)
    plt.ylabel('Mean F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename_prefix}classification_f1_comparison.png')
    plt.close()
    

    
    # Individual model plots
    for key, result in results.items():
        dataset_name_plot, model_name = key.split('_', 1)
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'{model_name} - Normalized Confusion Matrix ({dataset_name_plot})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{save_dir}/{key}_confusion_matrix.png')
        plt.close()
