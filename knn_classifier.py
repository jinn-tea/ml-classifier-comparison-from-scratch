"""
k-NN Classifier Implementation from Scratch

Implements k-Nearest Neighbors classifier with:
- Euclidean and Manhattan distance metrics
- Weighted voting based on distance
- Configurable k values
"""

import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean, cityblock


class KNNClassifier:
    """
    k-Nearest Neighbors Classifier implemented from scratch.
    """
    
    def __init__(self, k=5, distance_metric='euclidean', weighted=True):
        """
        Initialize k-NN Classifier.
        
        Parameters:
        -----------
        k : int, default=5
            Number of neighbors to consider
        distance_metric : str, default='euclidean'
            Distance metric to use ('euclidean' or 'manhattan')
        weighted : bool, default=True
            If True, use distance-weighted voting. If False, use majority voting.
        """
        self.k = k
        self.distance_metric = distance_metric.lower()
        self.weighted = weighted
        self.X_train = None
        self.y_train = None
        
        if self.distance_metric not in ['euclidean', 'manhattan']:
            raise ValueError("distance_metric must be 'euclidean' or 'manhattan'")
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points.
        
        Parameters:
        -----------
        x1 : array-like
            First point
        x2 : array-like
            Second point
            
        Returns:
        --------
        float
            Distance between x1 and x2
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:  # manhattan
            return np.sum(np.abs(x1 - x2))
    
    def fit(self, X, y):
        """
        Fit the k-NN classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
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
        predictions = []
        
        for sample in X:
            # Calculate distances to all training samples
            distances = [self._calculate_distance(sample, x) for x in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            
            if self.weighted:
                # Weighted voting: weights are inverse of distance
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                weights = [1 / (d + epsilon) for d in k_distances]
                
                # Calculate weighted vote for each class
                class_votes = {}
                for label, weight in zip(k_labels, weights):
                    if label not in class_votes:
                        class_votes[label] = 0
                    class_votes[label] += weight
                
                # Predict class with highest weighted vote
                prediction = max(class_votes, key=class_votes.get)
            else:
                # Majority voting
                prediction = Counter(k_labels).most_common(1)[0][0]
            
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
        unique_classes = np.unique(self.y_train)
        probabilities = []
        
        for sample in X:
            # Calculate distances to all training samples
            distances = [self._calculate_distance(sample, x) for x in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [self.y_train[i] for i in k_indices]
            
            if self.weighted:
                # Weighted probabilities
                epsilon = 1e-10
                weights = [1 / (d + epsilon) for d in k_distances]
                
                class_probs = {cls: 0.0 for cls in unique_classes}
                total_weight = sum(weights)
                
                for label, weight in zip(k_labels, weights):
                    class_probs[label] += weight
                
                # Normalize
                probs = [class_probs[cls] / total_weight for cls in unique_classes]
            else:
                # Majority voting probabilities
                vote_counts = Counter(k_labels)
                total_votes = len(k_labels)
                probs = [vote_counts.get(cls, 0) / total_votes for cls in unique_classes]
            
            probabilities.append(probs)
        
        return np.array(probabilities)
