"""
Evaluation Utilities

Provides functions for calculating metrics and generating visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)


def calculate_metrics(y_true, y_pred, y_proba=None, average='weighted'):
    """
    Calculate classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities (for ROC-AUC)
    average : str, default='weighted'
        Averaging strategy for multi-class metrics
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Calculate ROC-AUC if probabilities are provided and binary classification
    if y_proba is not None:
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
            else:
                # Multi-class: calculate for each class and average
                n_classes = len(np.unique(y_true))
                roc_aucs = []
                for i in range(n_classes):
                    y_true_binary = (y_true == np.unique(y_true)[i]).astype(int)
                    if len(np.unique(y_true_binary)) == 2:  # Check if class exists
                        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                        roc_aucs.append(auc(fpr, tpr))
                if roc_aucs:
                    metrics['roc_auc'] = np.mean(roc_aucs)
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', ax=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of classes
    title : str, default='Confusion Matrix'
        Plot title
    ax : matplotlib.axes, optional
        Axes to plot on
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    return ax


def plot_roc_curve(y_true, y_proba, class_names=None, title='ROC Curve', ax=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    class_names : list, optional
        Names of classes
    title : str, default='ROC Curve'
        Plot title
    ax : matplotlib.axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Multi-class: plot ROC for each class
        for i, cls in enumerate(unique_classes):
            y_true_binary = (y_true == cls).astype(int)
            if len(np.unique(y_true_binary)) == 2:
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                label = class_names[i] if class_names else f'Class {cls}'
                ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_learning_curve(train_scores, val_scores, param_values, param_name, 
                       title='Learning Curve', ax=None):
    """
    Plot learning curve.
    
    Parameters:
    -----------
    train_scores : array-like
        Training scores for different parameter values
    val_scores : array-like
        Validation scores for different parameter values
    param_values : array-like
        Parameter values
    param_name : str
        Name of the parameter
    title : str, default='Learning Curve'
        Plot title
    ax : matplotlib.axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_values, train_scores, 'o-', label='Training Score', linewidth=2)
    ax.plot(param_values, val_scores, 'o-', label='Validation Score', linewidth=2)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_feature_importance(importance_dict, title='Feature Importance', ax=None, top_n=10):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary mapping feature names to importance scores
    title : str, default='Feature Importance'
        Plot title
    ax : matplotlib.axes, optional
        Axes to plot on
    top_n : int, default=10
        Number of top features to display
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N features
    top_features = sorted_features[:top_n]
    features, importances = zip(*top_features)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.invert_yaxis()  # Top feature at top
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def print_metrics(metrics, model_name='Model'):
    """
    Print metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    model_name : str, default='Model'
        Name of the model
    """
    print(f"\n{model_name} Metrics:")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("-" * 40)
