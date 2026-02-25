# FinScore: Alternative Credit Scoring & Default Prediction

## 📌 Project Overview
Traditional credit scoring relies heavily on Bureau scores, often excluding "new-to-credit" or "thin-file" segments. This project develops a **Fintech-focused Credit Risk Model** that incorporates alternative data points—such as debt-to-income (DTI) ratios, employment stability, and inquiry frequency—to predict the Probability of Default ($PD$).

This framework is designed to mimic the assessment logic used by leading digital lenders like credit provider company to provide personal loans to underserved segments.

## 📊 Dataset & Reference Sources
The analysis utilizes a high-fidelity synthetic dataset modeled after the **LendingClub Open Data** and **DLAI (Digital Lenders Association of India)** industry benchmarks.
* **Target Segments:** Unsecured personal loans up to ₹10L.
* **Key Features:** Income stability, Credit utilization, DTI, and Bureau score.

## 🛠️ Methodology
1. **Feature Engineering:** - Created an `Income_Stability_Index` and `Installment_to_Income` ratio to assess repayment capacity beyond simple salary figures.
   - Developed a `Custom_Risk_Score` blending traditional bureau data with alternative financial health indicators.
2. **Class Imbalance Handling:** - Addressed the naturally low default rate (15%) by utilizing the `scale_pos_weight` hyperparameter within XGBoost to improve sensitivity toward high-risk borrowers.
3. **Advanced Modeling:** - Implemented an **XGBoost Classifier**, an industry-standard for non-linear credit risk modeling.

## 🚀 Key Results
- **Model Performance:** Achieved an **AUC-ROC score of 0.81**, indicating strong discriminative power.
- **Business Insights:** Identified "hidden gem" borrowers—users with average bureau scores but high repayment probability based on stable DTI and employment tenure.
- **Primary Risk Drivers:** Debt-to-Income (DTI), Annual Income, and recent Credit Inquiries.

## 📐 Mathematical Formulation
This model treats credit risk as a binary classification problem ($y \in \{0, 1\}$), where $1$ denotes default. The XGBoost objective function minimized is:
$$L(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)$$
Where $l$ is the differentiable convex loss function (logistic loss) and $\Omega$ penalizes the complexity of the model to prevent overfitting.
