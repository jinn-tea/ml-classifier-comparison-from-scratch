"""
Decision Tree Implementation from Scratch

Implements decision tree classifier with:
- Gini impurity and Entropy as split criteria
- Recursive binary splitting
- Configurable max depth
"""

import numpy as np
from collections import Counter


class DecisionTree:
    """
    Decision Tree Classifier implemented from scratch.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        """
        Initialize Decision Tree.
        
        Parameters:
        -----------
        max_depth : int or None, default=None
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum number of samples required to split a node
        criterion : str, default='gini'
            Splitting criterion ('gini' or 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion.lower()
        self.tree = None
        
        if self.criterion not in ['gini', 'entropy']:
            raise ValueError("criterion must be 'gini' or 'entropy'")
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity.
        
        Parameters:
        -----------
        y : array-like
            Target values
            
        Returns:
        --------
        float
            Gini impurity
        """
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        proportions = [count / len(y) for count in counts.values()]
        return 1 - sum(p ** 2 for p in proportions)
    
    def _entropy(self, y):
        """
        Calculate entropy.
        
        Parameters:
        -----------
        y : array-like
            Target values
            
        Returns:
        --------
        float
            Entropy
        """
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        proportions = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) if p > 0 else 0 for p in proportions)
    
    def _impurity(self, y):
        """
        Calculate impurity based on criterion.
        
        Parameters:
        -----------
        y : array-like
            Target values
            
        Returns:
        --------
        float
            Impurity value
        """
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        else:  # entropy
            return self._entropy(y)
    
    def _information_gain(self, y_parent, y_left, y_right):
        """
        Calculate information gain from a split.
        
        Parameters:
        -----------
        y_parent : array-like
            Target values of parent node
        y_left : array-like
            Target values of left child
        y_right : array-like
            Target values of right child
            
        Returns:
        --------
        float
            Information gain
        """
        parent_impurity = self._impurity(y_parent)
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n == 0:
            return 0
        
        weighted_impurity = (n_left / n) * self._impurity(y_left) + \
                           (n_right / n) * self._impurity(y_right)
        
        return parent_impurity - weighted_impurity
    
    def _find_best_split(self, X, y):
        """
        Find the best split for a node.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        dict
            Best split information
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try thresholds between consecutive values
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': best_gain
        }
    
    def _majority_class(self, y):
        """
        Get the majority class.
        
        Parameters:
        -----------
        y : array-like
            Target values
            
        Returns:
        --------
            Majority class label
        """
        counts = Counter(y)
        return counts.most_common(1)[0][0]
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target values
        depth : int, default=0
            Current depth of the tree
            
        Returns:
        --------
        dict
            Tree node
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (n_classes == 1):
            return {
                'leaf': True,
                'class': self._majority_class(y),
                'samples': n_samples
            }
        
        # Find best split
        best_split = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_split['gain'] == 0 or best_split['feature'] is None:
            return {
                'leaf': True,
                'class': self._majority_class(y),
                'samples': n_samples
            }
        
        # Split data
        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        right_mask = ~left_mask
        
        X_left = X[left_mask]
        y_left = y[left_mask]
        X_right = X[right_mask]
        y_right = y[right_mask]
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'gain': best_split['gain'],
            'left': left_child,
            'right': right_child,
            'samples': n_samples
        }
    
    def fit(self, X, y):
        """
        Fit the decision tree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, sample, node):
        """
        Predict class for a single sample.
        
        Parameters:
        -----------
        sample : array-like
            Sample to predict
        node : dict
            Current node in the tree
            
        Returns:
        --------
            Predicted class label
        """
        if node['leaf']:
            return node['class']
        
        if sample[node['feature']] <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
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
        predictions = [self._predict_sample(sample, self.tree) for sample in X]
        return np.array(predictions)
    
    def get_feature_importance(self, feature_names=None):
        """
        Calculate feature importance based on information gain.
        
        Parameters:
        -----------
        feature_names : array-like, optional
            Names of features
            
        Returns:
        --------
        dict
            Feature importance scores
        """
        def calculate_importance(node, total_samples):
            importance = {}
            if not node['leaf']:
                # Calculate importance as weighted gain
                feature_idx = node['feature']
                gain = node['gain']
                samples = node['samples']
                
                if feature_names:
                    feature_name = feature_names[feature_idx]
                else:
                    feature_name = f'feature_{feature_idx}'
                
                importance[feature_name] = gain * (samples / total_samples)
                
                # Recursively calculate for children
                if 'left' in node:
                    left_importance = calculate_importance(node['left'], total_samples)
                    for key, value in left_importance.items():
                        importance[key] = importance.get(key, 0) + value
                
                if 'right' in node:
                    right_importance = calculate_importance(node['right'], total_samples)
                    for key, value in right_importance.items():
                        importance[key] = importance.get(key, 0) + value
            
            return importance
        
        if self.tree is None:
            return {}
        
        total_samples = self.tree['samples']
        return calculate_importance(self.tree, total_samples)
