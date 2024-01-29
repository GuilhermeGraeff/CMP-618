# CMP-618

## Testes:
Systematicaly erasing features
F1 - Score de um SVM (Com e sem seleção de features)

## Algoritmos de seleção de Features

### Filters
minimum Redundancy Maximum Relevance Feature Selection (mRMR)  
chi²  
F-value  
mutual_info_classif  

### Embedded
Random Forest     - Feature importance  
Gradient Boosting - Feature importance  

### Sob a mesma rede  
Relevance Agregation  
SHAP  
LIME  
DeepLift  

### Configuração da rede:  


Toda essa seleção de features (top 10) será comparada com um SVM
-> SVM