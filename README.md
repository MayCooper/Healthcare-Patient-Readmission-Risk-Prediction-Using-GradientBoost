# Healthcare Patient Readmission Risk Prediction Using Gradient Boost

---

## Overview

This project implements a tuned Gradient Boosting Classifier to predict patient-related outcomes using structured healthcare records. By optimizing model parameters and evaluating key performance metrics, it identifies influential features and delivers actionable insights for healthcare cost management and outcome forecasting.

The analysis reveals how variables such as vitamin D levels, total charges, and patient age contribute to outcomes like readmission or risk categories. This enables data-driven decision-making in medical triage, preventive care, and health insurance underwriting.

---

## Project Objectives

- **Primary Goal**: Accurately classify medical outcomes using clinical and billing data.
- **Use Case**: Help healthcare professionals and insurance analysts identify high-risk patients and forecast medical cost trends.
- **Methodology**: Train and optimize a Gradient Boosting Classifier with cross-validation, followed by thorough performance evaluation using AUC-ROC, confusion matrix, and F1 Score.

---

## Research Question

Which factors are most predictive of patient outcomes, and how effectively can a Gradient Boosting model distinguish between different patient categories?

---

## Dataset Summary

The dataset includes over 3,100 rows with features such as medical charges, vitamin D levels, chronic condition flags, and demographic information. This blend of clinical, lifestyle, and financial data enables rich modeling of healthcare trends.

### Sample of the Dataset

| Total Medical Charges ($) | Vitamin D Level (ng/mL) | Days Admitted | Readmitted? | High Blood Pressure | Diabetes | Complication Risk | Annual Income ($) | Age (Years) |
|---------------------------|--------------------------|----------------|--------------|----------------------|----------|--------------------|--------------------|--------------|
| 3726.70                   | 19.14                    | 10.59          | No           | Yes                  | Yes      | Medium             | 86575.93           | 53           |
| 4193.19                   | 18.94                    | 15.13          | No           | Yes                  | No       | High               | 46805.99           | 51           |
| 2434.23                   | 18.06                    | 4.77           | No           | Yes                  | Yes      | Medium             | 14370.14           | 53           |
| 2127.83                   | 16.58                    | 1.71           | No           | No                   | No       | Medium             | 39741.49           | 78           |
| 2113.07                   | 17.44                    | 1.25           | No           | No                   | No       | Low                | 1209.56            | 22           |

---

## Model Optimization and Performance

### Hyperparameter Tuning

- Used `GridSearchCV` with 5-fold cross-validation to identify optimal model parameters.
- **Best Parameters**:
  ```python
  {'learning_rate': 0.0269, 'max_depth': 6, 'n_estimators': 207}
  ```
- **Best CV Accuracy**: 95.27%

The combination of a low learning rate and a moderate number of trees contributed to generalization and avoided overfitting.

---

### Final Model Metrics (Test Set)

| Metric     | Value   | Interpretation |
|------------|---------|----------------|
| Accuracy   | 0.9460  | 94.6% of total predictions were correct. |
| Precision  | 0.9097  | Of those predicted as positive, 91% were actually positive. |
| Recall     | 0.9469  | The model correctly identified ~95% of actual positive cases. |
| F1 Score   | 0.9279  | Balanced score accounting for both precision and recall. |
| AUC-ROC    | 0.9920  | Model is excellent at distinguishing between classes. |

These metrics paint a strong picture of the model’s performance. An accuracy of 94.6% shows that the classifier is correct the vast majority of the time, but accuracy alone isn’t enough in imbalanced or high-risk domains like healthcare. The **precision** of 0.91 indicates that when the model predicts a patient as high-risk or positive, it is usually correct—minimizing false alarms. At the same time, a **recall** of nearly 95% means it’s also catching most of the actual positive cases, ensuring that at-risk patients are not overlooked. The **F1 Score**, which balances both precision and recall, reflects the model’s overall robustness. Most notably, the **AUC-ROC of 0.9920** signals that the classifier is almost perfect in separating the two classes, regardless of any imbalance. Taken together, these results validate the model as a reliable and effective tool for classification tasks where both accuracy and caution are critical.

---

### Confusion Matrix

|            | Predicted: No | Predicted: Yes |
|------------|----------------|----------------|
| Actual: No | 1197           | 69             |
| Actual: Yes| 39             | 695            |

![image](https://github.com/user-attachments/assets/b9ec68b0-13b2-4afc-a2f5-7e5bc1e7f7cc)

The confusion matrix provides a clear visualization of the model’s classification performance on the test set. Out of all actual negative cases (patients who do not belong to the positive class), the model correctly identified 1,197 of them, misclassifying only 69. This indicates a high **true negative rate**, meaning the model is reliable at filtering out false alarms.

On the other side, for actual positive cases, it correctly identified 695 patients while misclassifying 39 as negative. This translates to a **high true positive rate (recall)** of approximately 94.7%, meaning the model is highly sensitive to identifying positive outcomes, which is crucial in medical contexts where missing a high-risk case can have serious consequences.

The relatively low number of **false positives (69)** and **false negatives (39)** suggests a balanced model that does not heavily skew toward over-predicting or under-predicting any class. This balance contributes to the strong F1 score and indicates that the model performs consistently across both classes.


---

### ROC Curve

![image](https://github.com/user-attachments/assets/386b6113-312f-4219-a258-03768c16331e)

- AUC of **0.9920** suggests near-perfect separation between classes, making this model suitable for real-world deployment.

---

### Feature Importance

![image](https://github.com/user-attachments/assets/aae69c30-f99b-4c10-a2dd-9faf1a323efb)

- `TotalCharge` dominates the importance ranking.
- `Initial_days` and `VitD_levels` follow, suggesting clinical and economic indicators are both influential.

---

## Key Insights

- The model achieves **high precision and recall**, supporting its use in sensitive healthcare applications.
- `TotalCharge` may reflect treatment complexity or underlying risk.
- Variables like vitamin D levels or hospitalization duration could act as early markers for future complications.
- Feature importance helps direct future data collection priorities or clinical screenings.

---

## Tools and Technologies

- **Python** – Data analysis and modeling
- **Pandas, NumPy** – Data preprocessing
- **Scikit-learn** – Model building and evaluation
- **Matplotlib, Seaborn** – Visual analysis
- **GridSearchCV** – Hyperparameter tuning

---

## Conclusion

This classification project highlights the power and flexibility of gradient boosting for uncovering complex patterns in healthcare data. By tuning and evaluating the model with real-world medical and demographic features, we demonstrated that even a relatively simple feature set—when paired with a strong algorithm—can deliver highly accurate predictions with excellent precision and recall.

The model’s strong AUC-ROC score suggests it's not just fitting the training data but is genuinely effective at distinguishing between different patient risk profiles. The clear ranking of feature importance also helps explain the results, making this approach transparent and potentially valuable for healthcare providers, analysts, and decision-makers.

That said, this work represents just one step in a broader effort. Future improvements could include incorporating time-series data such as treatment progression, lab results, or patient vitals over time. Adding these dynamic components could help the model adapt to more nuanced patient scenarios and further enhance its predictive power. Additionally, evaluating fairness across different patient subgroups would be a crucial step before deploying the model in a clinical or operational setting.

Overall, this project provides a strong foundation for building intelligent, data-driven tools in healthcare analytics—tools that not only predict but inform and guide.
