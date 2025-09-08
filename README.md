# RISKON  
**Response Informed Scalar Kinetics for Optimal Nudging**  

🔗 **Website**: [https://riskontech.github.io/](https://riskontech.github.io/)  

RISKON is a **dynamic, transparent, and regulatory-compliant credit risk assessment framework** designed for the Indian financial ecosystem. It leverages advanced feature engineering, Weight of Evidence (WoE) transformations, Information Value (IV) selection, unsupervised cohort discovery, and dynamic Ordinary Differential Equation (ODE)-based modelling to deliver personalised and auditable credit intelligence.  

The system was built using the Home Credit Default Risk Dataset as a proxy for real-world lending portfolios.  

---

## Overview
RISKON was developed to address limitations in traditional credit scoring by:  
- Incorporating **dynamic behavioural signals** into risk scoring.  
- Enabling **personalised risk equations** based on borrower archetypes.  
- Maintaining **glass-box transparency** with WoE, IV, and interpretable regression models.  
- Ensuring **scalability** for large datasets using efficient computation (Dask, Pandas, Scikit-learn, SciPy).  

---

## Key Features
- **Unified Time-Series Dataset** combining static applicant attributes and dynamic behavioural features.  
- **WoE Transformation & IV-based Feature Selection** for robust, interpretable feature engineering.  
- **K-Means Clustering** to segment borrowers into distinct financial archetypes.  
- **ODE-based Modelling** for capturing dynamic credit risk evolution over time.  
- **ElasticNetCV Regression** for cohort-specific master risk equations.  
- **Auditable Pipeline** for regulatory alignment and reproducibility.  

