import shap  
import matplotlib.pyplot as plt 


def explainer(model, X): 
    
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X)
    
    shap.summary_plot(shap_values, X, plot_type="bar") 
    plt.tight_layout() 
    plt.savefig("shap_summary_plot.png") 
    plt.show() 
    
    return shap_explainer, shap_values 