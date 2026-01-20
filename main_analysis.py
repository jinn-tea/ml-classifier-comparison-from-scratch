"""
Main Analysis Script for Classification Algorithms Comparison

This script performs:
- Model training with hyperparameter tuning
- Evaluation and comparison of k-NN, Random Forest, and Naive Bayes
- Visualization of results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from knn_classifier import KNNClassifier
from random_forest import RandomForest
from naive_bayes import NaiveBayes
from evaluation_utils import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_learning_curve, plot_feature_importance, print_metrics
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_preprocessed_data():
    """Load preprocessed data from pickle file."""
    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def tune_knn(X_train, y_train, X_val, y_val, k_values=[1, 3, 5, 7, 10]):
    """
    Tune k-NN hyperparameters.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_val, y_val : validation data
    k_values : list of k values to try
    
    Returns:
    --------
    dict : best parameters and results
    """
    print("="*60)
    print("Tuning k-NN Classifier")
    print("="*60)
    
    results = []
    
    for metric in ['euclidean', 'manhattan']:
        for k in k_values:
            for weighted in [True, False]:
                # Train model
                knn = KNNClassifier(k=k, distance_metric=metric, weighted=weighted)
                knn.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = knn.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                
                results.append({
                    'k': k,
                    'metric': metric,
                    'weighted': weighted,
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score']
                })
                
                print(f"k={k}, metric={metric}, weighted={weighted}: "
                      f"Accuracy={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1_score']:.4f}")
    
    # Find best configuration
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1_score'].idxmax()
    best_params = results_df.iloc[best_idx].to_dict()
    
    print(f"\nBest k-NN configuration:")
    print(f"k={best_params['k']}, metric={best_params['metric']}, "
          f"weighted={best_params['weighted']}")
    print(f"Validation F1-Score: {best_params['f1_score']:.4f}\n")
    
    return best_params, results_df


def tune_random_forest(X_train, y_train, X_val, y_val, max_depths=[3, 5, 7, 10, None]):
    """
    Tune Random Forest hyperparameters.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_val, y_val : validation data
    max_depths : list of max_depth values to try
    
    Returns:
    --------
    dict : best parameters and results
    """
    print("="*60)
    print("Tuning Random Forest Classifier")
    print("="*60)
    
    results = []
    
    for criterion in ['gini', 'entropy']:
        for max_depth in max_depths:
            # Train model
            rf = RandomForest(
                n_trees=5,
                max_depth=max_depth,
                criterion=criterion,
                random_state=42
            )
            rf.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = rf.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred)
            
            results.append({
                'max_depth': max_depth if max_depth else 'None',
                'criterion': criterion,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score']
            })
            
            print(f"max_depth={max_depth}, criterion={criterion}: "
                  f"Accuracy={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}")
    
    # Find best configuration
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1_score'].idxmax()
    best_params = results_df.iloc[best_idx].to_dict()
    
    print(f"\nBest Random Forest configuration:")
    print(f"max_depth={best_params['max_depth']}, criterion={best_params['criterion']}")
    print(f"Validation F1-Score: {best_params['f1_score']:.4f}\n")
    
    return best_params, results_df


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """
    Evaluate a model on train, validation, and test sets.
    
    Parameters:
    -----------
    model : trained model
    X_train, y_train, X_val, y_val, X_test, y_test : data splits
    model_name : str, name of the model
    
    Returns:
    --------
    dict : evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # Train set
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else None
    results['train'] = calculate_metrics(y_train, y_pred_train, y_proba_train)
    
    # Validation set
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
    results['val'] = calculate_metrics(y_val, y_pred_val, y_proba_val)
    
    # Test set
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    results['test'] = calculate_metrics(y_test, y_pred_test, y_proba_test)
    
    # Print results
    print_metrics(results['train'], f"{model_name} (Train)")
    print_metrics(results['val'], f"{model_name} (Validation)")
    print_metrics(results['test'], f"{model_name} (Test)")
    
    # Store predictions for visualization
    results['predictions'] = {
        'train': y_pred_train,
        'val': y_pred_val,
        'test': y_pred_test
    }
    results['probabilities'] = {
        'train': y_proba_train,
        'val': y_proba_val,
        'test': y_proba_test
    }
    
    return results


def plot_comparison(models_results, model_names):
    """
    Plot comparison of all models.
    
    Parameters:
    -----------
    models_results : dict of results for each model
    model_names : list of model names
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        test_scores = [models_results[name]['test'][metric] for name in model_names]
        
        bars = ax.bar(model_names, test_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison (Test Set)')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main analysis function."""
    print("Loading preprocessed data...")
    data = load_preprocessed_data()
    
    X_train = data['X_train_scaled']
    X_val = data['X_val_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print(f"Data loaded successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # Store results
    all_results = {}
    models = {}
    
    # ========== k-NN Tuning and Training ==========
    best_knn_params, knn_tuning_results = tune_knn(X_train, y_train, X_val, y_val)
    
    knn = KNNClassifier(
        k=int(best_knn_params['k']),
        distance_metric=best_knn_params['metric'],
        weighted=best_knn_params['weighted']
    )
    knn.fit(X_train, y_train)
    models['k-NN'] = knn
    
    all_results['k-NN'] = evaluate_model(
        knn, X_train, y_train, X_val, y_val, X_test, y_test, 'k-NN'
    )
    
    # Learning curve for k-NN
    k_values = [1, 3, 5, 7, 10]
    knn_train_scores = []
    knn_val_scores = []
    
    for k in k_values:
        knn_temp = KNNClassifier(k=k, distance_metric=best_knn_params['metric'],
                                weighted=best_knn_params['weighted'])
        knn_temp.fit(X_train, y_train)
        
        train_pred = knn_temp.predict(X_train)
        val_pred = knn_temp.predict(X_val)
        
        knn_train_scores.append(calculate_metrics(y_train, train_pred)['accuracy'])
        knn_val_scores.append(calculate_metrics(y_val, val_pred)['accuracy'])
    
    plt.figure(figsize=(10, 6))
    plot_learning_curve(knn_train_scores, knn_val_scores, k_values, 'k',
                       'k-NN Learning Curve')
    plt.savefig('knn_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== Random Forest Tuning and Training ==========
    best_rf_params, rf_tuning_results = tune_random_forest(X_train, y_train, X_val, y_val)
    
    max_depth = None if best_rf_params['max_depth'] == 'None' else int(best_rf_params['max_depth'])
    rf = RandomForest(
        n_trees=5,
        max_depth=max_depth,
        criterion=best_rf_params['criterion'],
        random_state=42
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    all_results['Random Forest'] = evaluate_model(
        rf, X_train, y_train, X_val, y_val, X_test, y_test, 'Random Forest'
    )
    
    # Feature importance for Random Forest
    rf_importance = rf.get_feature_importance(feature_names)
    plt.figure(figsize=(10, 6))
    plot_feature_importance(rf_importance, 'Random Forest Feature Importance')
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Learning curve for Random Forest
    max_depths = [3, 5, 7, 10, 15]
    rf_train_scores = []
    rf_val_scores = []
    
    for md in max_depths:
        rf_temp = RandomForest(n_trees=5, max_depth=md,
                              criterion=best_rf_params['criterion'],
                              random_state=42)
        rf_temp.fit(X_train, y_train)
        
        train_pred = rf_temp.predict(X_train)
        val_pred = rf_temp.predict(X_val)
        
        rf_train_scores.append(calculate_metrics(y_train, train_pred)['accuracy'])
        rf_val_scores.append(calculate_metrics(y_val, val_pred)['accuracy'])
    
    plt.figure(figsize=(10, 6))
    plot_learning_curve(rf_train_scores, rf_val_scores, max_depths, 'Max Depth',
                       'Random Forest Learning Curve')
    plt.savefig('rf_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== Naive Bayes Training ==========
    print("="*60)
    print("Training Naive Bayes Classifier")
    print("="*60)
    
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    
    all_results['Naive Bayes'] = evaluate_model(
        nb, X_train, y_train, X_val, y_val, X_test, y_test, 'Naive Bayes'
    )
    
    # ========== Visualization ==========
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_names = ['k-NN', 'Random Forest', 'Naive Bayes']
    
    for idx, (name, ax) in enumerate(zip(model_names, axes)):
        y_pred = all_results[name]['predictions']['test']
        plot_confusion_matrix(y_test, y_pred, 
                            title=f'{name} Confusion Matrix',
                            ax=ax)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, ax) in enumerate(zip(model_names, axes)):
        y_proba = all_results[name]['probabilities']['test']
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba,
                          title=f'{name} ROC Curve',
                          ax=ax)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Model comparison
    plot_comparison(all_results, model_names)
    
    # ========== Summary Table ==========
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    summary_data = []
    for name in model_names:
        test_metrics = all_results[name]['test']
        summary_data.append({
            'Model': name,
            'Accuracy': f"{test_metrics['accuracy']:.4f}",
            'Precision': f"{test_metrics['precision']:.4f}",
            'Recall': f"{test_metrics['recall']:.4f}",
            'F1-Score': f"{test_metrics['f1_score']:.4f}",
            'ROC-AUC': f"{test_metrics.get('roc_auc', 'N/A')}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv('results_summary.csv', index=False)
    print("\nResults saved to 'results_summary.csv'")
    
    return models, all_results


if __name__ == '__main__':
    models, results = main()
