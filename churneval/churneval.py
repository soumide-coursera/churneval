# -*- coding: utf-8 -*-
"""
@author: Soumi De
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

def get_performance_metrics(model_name: str, true_class: pd.DataFrame, predicted_class: pd.arrays, predicted_probs: pd.arrays):
    
    test_index = true_class.index
    conf_mat = confusion_matrix(predicted_class, true_class, labels=[1,0])
    tdl = pd.concat([pd.DataFrame(predicted_probs,index=test_index, columns=['predicted_probs']),
                        pd.DataFrame(predicted_class,index=test_index, columns=['predicted_class']),
                        pd.DataFrame(true_class,index=test_index)], 
                    axis = 1, ignore_index=False).sort_values('predicted_probs', ascending = False)

    a = tdl.head(round(0.10*tdl.shape[0]))

    tdl_metric_num = a.query('y == 1')['y'].sum()/a.shape[0]
    tdl_metric_den = sum(conf_mat[:,0])/conf_mat.sum()
    tdl = tdl_metric_num/tdl_metric_den

    roc_value = roc_auc_score(true_class, predicted_probs) 
    accuracy = metrics.accuracy_score(true_class, predicted_class)
    precision = metrics.precision_score(true_class, predicted_class)
    sensitivity = metrics.recall_score(true_class, predicted_class)
    specificity = conf_mat[1,1]/sum(conf_mat[:,1])
    f1_score = metrics.f1_score(true_class, predicted_class)
    perf = pd.DataFrame()
    perf = perf.append({'Model_Name': model_name, 'Accuracy': accuracy, 'Confusion Matrix': conf_mat,
                        'Precision': precision, 'Sensitivity': sensitivity, 
                        'Specificity': specificity, 'F1-score': f1_score,'ROC_score':roc_value,
                        "Top_dec_lift":tdl}, ignore_index=True)
    
    return(perf)

# Calculates top decile lift of a model
def top_decile_lift(true_class: pd.DataFrame, predicted_class: pd.arrays, predicted_probs: pd.arrays):
    
    test_index = true_class.index
    conf_mat = confusion_matrix(predicted_class, true_class, labels=[1,0])
    tdl = pd.concat([pd.DataFrame(predicted_probs,index=test_index, columns=['predicted_probs']),
                        pd.DataFrame(predicted_class,index=test_index, columns=['predicted_class']),
                        pd.DataFrame(true_class,index=test_index)], 
                    axis = 1, ignore_index=False).sort_values('predicted_probs', ascending = False)

    a = tdl.head(round(0.10*tdl.shape[0]))

    tdl_metric_num = a.query('y == 1')['y'].sum()/a.shape[0]
    tdl_metric_den = sum(conf_mat[:,0])/conf_mat.sum()
    tdl = round(tdl_metric_num/tdl_metric_den,3)
    
    return(tdl)


# Calculates lift curve of a model
def lift_curve(true_class: pd.DataFrame, predicted_class: pd.arrays, predicted_probs: pd.arrays):
    
    test_index = true_class.index
    conf_mat = confusion_matrix(predicted_class, true_class, labels=[1,0])
    tdl = pd.concat([pd.DataFrame(predicted_probs,index=test_index, columns=['predicted_probs']),
                        pd.DataFrame(predicted_class,index=test_index, columns=['predicted_class']),
                        pd.DataFrame(true_class,index=test_index)], 
                    axis = 1, ignore_index=False).sort_values('predicted_probs', ascending = False)
    
    lift = pd.DataFrame(columns=['prop', 'lift','baseline'])
    tdl_metric_den = sum(conf_mat[:,0])/conf_mat.sum()
    
    for i in range(10,105,5):
        a = tdl.head(round((i/100)*tdl.shape[0]))
        tdl_metric_num = a.query('y == 1')['y'].sum()/a.shape[0]
        l = round(tdl_metric_num/tdl_metric_den,3)
        lift = lift.append({'prop': i/100, 'lift': l, 'baseline': 1}, ignore_index=True)
    
    plt.figure()
    plt.plot(lift.prop,lift.lift, 'g-o', label = 'Lift')
    plt.plot(lift.prop, lift.baseline, color = 'blue',ls = '--', label = 'Baseline')
    plt.xlabel('Proportion of Data')
    plt.ylabel('Lift')
    plt.xlim([0.1, 1.0])
    plt.title("Lift Curve")
    plt.legend()
    plt.show()
