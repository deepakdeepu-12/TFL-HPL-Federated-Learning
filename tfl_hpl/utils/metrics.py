"""Evaluation Metrics for TFL-HPL

Computes accuracy, precision, recall, F1, confusion matrix, etc.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Evaluation metrics container"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None


class MetricsComputer:
    """Compute evaluation metrics"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       y_pred_proba: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """Compute standard evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for AUC)
            
        Returns:
            EvaluationMetrics object
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        auc = None
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            confusion_matrix=cm
        )
    
    @staticmethod
    def compute_attack_detection_metrics(y_true: np.ndarray,
                                        y_pred: np.ndarray) -> Dict:
        """Compute attack detection metrics
        
        Args:
            y_true: Ground truth (0=benign, 1=attack)
            y_pred: Predictions
            
        Returns:
            Dictionary with detection metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        }
