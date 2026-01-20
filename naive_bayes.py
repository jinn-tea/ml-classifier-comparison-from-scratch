"""
Naive Bayes Implementation from Scratch

Implements Naive Bayes classifier with:
- Gaussian Naive Bayes for continuous features
- Frequency-based probability estimation for categorical features
"""

import numpy as np
from collections import Counter, defaultdict


class NaiveBayes:
    """
    Naive Bayes Classifier implemented from scratch.
    Supports both continuous (Gaussian) and categorical features.
    """
    
    def __init__(self, feature_types=None):
        """
        Initialize Naive Bayes classifier.
        
        Parameters:
        -----------
        feature_types : list of str, optional
            List specifying type of each feature: 'continuous' or 'categorical'
            If None, all features are assumed to be continuous
        """
        self.feature_types = feature_types
        self.classes_ = None
        self.class_prior_ = None
        self.feature_params_ = None
    
    def _calculate_gaussian_params(self, X):
        """
        Calculate mean and standard deviation for Gaussian distribution.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,)
            Feature values for one feature
            
        Returns:
        --------
        tuple
            (mean, std) for Gaussian distribution
        """
        mean = np.mean(X)
        std = np.std(X)
        # Add small epsilon to avoid zero std
        std = max(std, 1e-10)
        return mean, std
    
    def _gaussian_pdf(self, x, mean, std):
        """
        Calculate Gaussian probability density function.
        
        Parameters:
        -----------
        x : float
            Feature value
        mean : float
            Mean of Gaussian distribution
        std : float
            Standard deviation of Gaussian distribution
            
        Returns:
        --------
        float
            Probability density
        """
        coefficient = 1 / (std * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mean) / std) ** 2
        return coefficient * np.exp(exponent)
    
    def _calculate_categorical_params(self, X):
        """
        Calculate probability distribution for categorical feature.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,)
            Feature values for one feature
            
        Returns:
        --------
        dict
            Probability for each category (with Laplace smoothing)
        """
        counts = Counter(X)
        total = len(X)
        
        # Laplace smoothing: add 1 to each count
        unique_values = set(X)
        probabilities = {}
        
        for value in unique_values:
            # Laplace smoothing: (count + 1) / (total + n_unique)
            probabilities[value] = (counts[value] + 1) / (total + len(unique_values))
        
        return probabilities
    
    def _categorical_prob(self, x, prob_dist):
        """
        Get probability for a categorical value.
        
        Parameters:
        -----------
        x : any
            Feature value
        prob_dist : dict
            Probability distribution for categories
            
        Returns:
        --------
        float
            Probability
        """
        if x in prob_dist:
            return prob_dist[x]
        else:
            # If value not seen in training, use Laplace smoothing
            return 1 / (sum(prob_dist.values()) * len(prob_dist) + 1)
    
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Determine feature types if not provided
        if self.feature_types is None:
            # Auto-detect: assume continuous for numeric, categorical for non-numeric
            self.feature_types = []
            for i in range(n_features):
                if np.issubdtype(X[:, i].dtype, np.number):
                    # Check if values look discrete (few unique values relative to samples)
                    unique_vals = len(np.unique(X[:, i]))
                    if unique_vals < 10 and unique_vals < len(X) * 0.1:
                        self.feature_types.append('categorical')
                    else:
                        self.feature_types.append('continuous')
                else:
                    self.feature_types.append('categorical')
        
        # Calculate class prior probabilities
        class_counts = Counter(y)
        total_samples = len(y)
        self.class_prior_ = {cls: count / total_samples 
                            for cls, count in class_counts.items()}
        
        # Calculate feature parameters for each class
        self.feature_params_ = {}
        
        for cls in self.classes_:
            X_class = X[y == cls]
            self.feature_params_[cls] = []
            
            for feature_idx in range(n_features):
                feature_values = X_class[:, feature_idx]
                feature_type = self.feature_types[feature_idx]
                
                if feature_type == 'continuous':
                    # Gaussian parameters
                    mean, std = self._calculate_gaussian_params(feature_values)
                    self.feature_params_[cls].append({
                        'type': 'gaussian',
                        'mean': mean,
                        'std': std
                    })
                else:  # categorical
                    # Categorical probability distribution
                    prob_dist = self._calculate_categorical_params(feature_values)
                    self.feature_params_[cls].append({
                        'type': 'categorical',
                        'prob_dist': prob_dist
                    })
        
        return self
    
    def _calculate_likelihood(self, x, feature_params):
        """
        Calculate likelihood for a single feature value.
        
        Parameters:
        -----------
        x : any
            Feature value
        feature_params : dict
            Parameters for the feature distribution
            
        Returns:
        --------
        float
            Likelihood (probability)
        """
        if feature_params['type'] == 'gaussian':
            return self._gaussian_pdf(x, feature_params['mean'], feature_params['std'])
        else:  # categorical
            return self._categorical_prob(x, feature_params['prob_dist'])
    
    def _calculate_posterior(self, X, cls):
        """
        Calculate posterior probability for a class.
        
        Parameters:
        -----------
        X : array-like of shape (n_features,)
            Sample features
        cls : any
            Class label
            
        Returns:
        --------
        float
            Log posterior probability
        """
        # Start with log prior
        log_posterior = np.log(self.class_prior_[cls])
        
        # Add log likelihood for each feature (Naive Bayes assumption: independence)
        for feature_idx, feature_value in enumerate(X):
            feature_params = self.feature_params_[cls][feature_idx]
            likelihood = self._calculate_likelihood(feature_value, feature_params)
            
            # Use log to avoid numerical underflow
            log_likelihood = np.log(max(likelihood, 1e-10))
            log_posterior += log_likelihood
        
        return log_posterior
    
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
            # Calculate posterior for each class
            posteriors = {cls: self._calculate_posterior(sample, cls) 
                         for cls in self.classes_}
            
            # Predict class with highest posterior
            prediction = max(posteriors, key=posteriors.get)
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
        probabilities = []
        
        for sample in X:
            # Calculate log posteriors for each class
            log_posteriors = {cls: self._calculate_posterior(sample, cls) 
                             for cls in self.classes_}
            
            # Convert to probabilities using log-sum-exp trick for numerical stability
            log_posterior_values = list(log_posteriors.values())
            max_log_posterior = max(log_posterior_values)
            
            # Normalize
            exp_log_posteriors = {cls: np.exp(log_p - max_log_posterior) 
                                 for cls, log_p in log_posteriors.items()}
            sum_exp = sum(exp_log_posteriors.values())
            
            probs = [exp_log_posteriors[cls] / sum_exp for cls in self.classes_]
            probabilities.append(probs)
        
        return np.array(probabilities)
