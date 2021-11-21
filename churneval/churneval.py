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

    tdl_metric_num = a.iloc[: , 2].sum()/a.shape[0]
    tdl_metric_den = sum(conf_mat[:,0])/conf_mat.sum()
    lift = round(tdl_metric_num/tdl_metric_den,3)

    roc_value = round(roc_auc_score(true_class, predicted_probs),3) 
    accuracy = round(metrics.accuracy_score(true_class, predicted_class),3)
    precision = round(metrics.precision_score(true_class, predicted_class),3)
    sensitivity = round(metrics.recall_score(true_class, predicted_class),3)
    specificity = round(conf_mat[1,1]/sum(conf_mat[:,1]),3)
    f1_score = round(metrics.f1_score(true_class, predicted_class),3)
    perf = pd.DataFrame(columns=['Model_Name','Accuracy','Confusion Matrix','Precision',
                                'Sensitivity','Specificity','F1-score','ROC_score','Top_dec_lift'])
    perf = perf.append({'Model_Name': model_name, 'Accuracy': accuracy, 'Confusion Matrix': conf_mat,
                        'Precision': precision, 'Sensitivity': sensitivity, 
                        'Specificity': specificity, 'F1-score': f1_score,'ROC_score':roc_value,
                        "Top_dec_lift":lift}, ignore_index=True)
    
    return(perf)

# Calculates top decile lift of a model
def top_decile_lift(true_class: pd.DataFrame, predicted_probs: pd.arrays):
    
    test_index = true_class.index

    tdl = pd.concat([pd.DataFrame(predicted_probs,index=test_index, columns=['predicted_probs']),
                        pd.DataFrame(true_class,index=test_index)], 
                    axis = 1, ignore_index=False).sort_values('predicted_probs', ascending = False)

    a = tdl.head(round(0.10*tdl.shape[0]))

    tdl_metric_num = a.iloc[: , 1].sum()/a.shape[0]
    tdl_metric_den = true_class.iloc[:,0].sum()/true_class.shape[0]
    lift = round(tdl_metric_num/tdl_metric_den,3)
    
    return(lift)


# Calculates lift curve of a model
def lift_curve(true_class: pd.DataFrame, predicted_probs: pd.arrays):
    
    test_index = true_class.index
    
    tdl = pd.concat([pd.DataFrame(predicted_probs,index=test_index, columns=['predicted_probs']),
                        pd.DataFrame(true_class,index=test_index)], 
                    axis = 1, ignore_index=False).sort_values('predicted_probs', ascending = False)
    
    lift = pd.DataFrame(columns=['prop', 'lift','baseline'])
    tdl_metric_den = true_class.iloc[:,0].sum()/true_class.shape[0]
    
    for i in range(10,105,5):
        a = tdl.head(round((i/100)*tdl.shape[0]))
        tdl_metric_num = a.iloc[: , 1].sum()/a.shape[0]
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
    

# Calculates gain curve of a model
def gain_curve(true_class: pd.DataFrame, predicted_probs: pd.arrays):
    
    test_index = true_class.index
    
    tdl = pd.concat([pd.DataFrame(predicted_probs,index=test_index, columns=['predicted_probs']),
                        pd.DataFrame(true_class,index=test_index)], 
                    axis = 1, ignore_index=False).sort_values('predicted_probs', ascending = False)
    
    gain = pd.DataFrame(columns=['prop', 'gain','random'])
    tdl_metric_den = true_class.iloc[:,0].sum()
    
    for i in range(0,105,5):
        a = tdl.head(round((i/100)*tdl.shape[0]))
        tdl_metric_num = a.iloc[: , 1].sum()
        l = round(tdl_metric_num/tdl_metric_den,3)
        gain = gain.append({'prop': i/100, 'gain': l, 'random': i/100}, ignore_index=True)
    
    plt.figure()
    plt.plot(gain.prop,gain.gain, 'g-o', label = 'Class = 1')
    plt.plot(gain.prop, gain.random, color = 'blue',ls = '--', label = 'Random')
    plt.xlabel('Proportion of Data')
    plt.ylabel('Gain')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Gain Curve")
    plt.legend()
    plt.show()
