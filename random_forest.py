"""
Random Forest Implementation from Scratch

Implements Random Forest classifier with:
- Bootstrap sampling (bagging)
- Multiple decision trees
- Majority voting for predictions
"""

import numpy as np
from decision_tree import DecisionTree
from collections import Counter


class RandomForest:
    """
    Random Forest Classifier implemented from scratch.
    """
    
    def __init__(self, n_trees=5, max_depth=None, min_samples_split=2, 
                 criterion='gini', random_state=None):
        """
        Initialize Random Forest.
        
        Parameters:
        -----------
        n_trees : int, default=5
            Number of trees in the forest
        max_depth : int or None, default=None
            Maximum depth of each tree
        min_samples_split : int, default=2
            Minimum number of samples required to split a node
        criterion : str, default='gini'
            Splitting criterion ('gini' or 'entropy')
        random_state : int or None, default=None
            Random seed for reproducibility
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        tuple
            Bootstrap sampled X and y
        """
        n_samples = len(y)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        Fit the Random Forest.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        
        self.trees = []
        
        for i in range(self.n_trees):
            # Create bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # Train decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples using majority voting.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for sample in X:
            # Get predictions from all trees
            tree_predictions = [tree.predict([sample])[0] for tree in self.trees]
            
            # Majority voting
            prediction = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        X = np.array(X)
        all_classes = set()
        
        # Collect all unique classes from training
        for tree in self.trees:
            if hasattr(tree, 'y_train'):
                all_classes.update(tree.y_train)
        
        if not all_classes:
            # If we don't have access to classes, use predictions to infer
            sample_preds = self.predict(X)
            all_classes = set(sample_preds)
        
        all_classes = sorted(list(all_classes))
        n_classes = len(all_classes)
        n_samples = len(X)
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, sample in enumerate(X):
            # Get predictions from all trees
            tree_predictions = [tree.predict([sample])[0] for tree in self.trees]
            
            # Calculate probability as fraction of trees voting for each class
            for j, cls in enumerate(all_classes):
                probabilities[i, j] = tree_predictions.count(cls) / len(self.trees)
        
        return probabilities
    
    def get_feature_importance(self, feature_names=None):
        """
        Calculate feature importance by averaging across all trees.
        
        Parameters:
        -----------
        feature_names : array-like, optional
            Names of features
            
        Returns:
        --------
        dict
            Average feature importance scores
        """
        all_importances = []
        
        for tree in self.trees:
            tree_importance = tree.get_feature_importance(feature_names)
            all_importances.append(tree_importance)
        
        # Average importances across all trees
        avg_importance = {}
        for importance_dict in all_importances:
            for feature, value in importance_dict.items():
                avg_importance[feature] = avg_importance.get(feature, 0) + value
        
        # Normalize
        n_trees = len(self.trees)
        for feature in avg_importance:
            avg_importance[feature] /= n_trees
        
        return avg_importance
