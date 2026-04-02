# Data Validation and Exploration Report

**Date:** 02/04/2026  
**Datasets Analyzed:** Diabetes Prediction, BRFSS 2015, BRFSS 2021  
**Objective:** Validate schema, confirm data completeness, check data types, and enforce categorical/numerical business rules.

---

## 1. Executive Summary
* **Diabetes Prediction Dataset:** **FAILED** validation. 0 missing values found. 3,854 duplicates identified. Failed categorical checks on the `smoking_history` and `gender` columns.
* **BRFSS 2015 Dataset:** **PASSED** validation. 0 missing values found. 24,206 duplicates identified. All data constraints and schemas passed perfectly.
* **BRFSS 2021 Dataset:** **FAILED** validation. 0 missing values found. 13,135 duplicates identified. Failed categorical check on the `Income` column due to updated category codes.

---

## 2. Dataset Details and Quality Logs

### Dataset A: Diabetes Prediction Dataset

**Initial Exploration Metrics**
> **Rows:** 100,000     
> **Columns:** 9    
> **Duplicates:** 3,854  
> **Missing Values:** 0 
> 
> **Data Types:** > int64: 4 columns -> `['hypertension', 'heart_disease', 'blood_glucose_level', 'diabetes']`    
> float64: 3 columns -> `['age', 'bmi', 'HbA1c_level']`  
> object: 2 columns -> `['gender', 'smoking_history']`    

**Rule-Based Validation**
> **Overall Result:** FAILED
> 
> **Failed Expectations Log:**
> * **[FAIL]** Categorical Value Set Match (`expect_column_values_to_be_in_set`)
>   * **Column:** `smoking_history`
>   * **Issues:** 4,004 unexpected values found (4.004% of total rows).
>   * **Sample:** `ever`
> 
> * **[FAIL]** Categorical Value Set Match (`expect_column_values_to_be_in_set`)
>   * **Column:** `gender`
>   * **Issues:** 18 unexpected values found (0.018% of total rows).
>   * **Sample:** `Other`

---

### Dataset B: Diabetes BRFSS 2015

**Initial Exploration Metrics**
> **Rows:** 253,680     
> **Columns:** 22      
> **Duplicates:** 24,206      
> **Missing Values:** 0 
> 
> **Data Types:** > float64: 22 columns -> `['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']`

**Rule-Based Validation**
> **Overall Result:** PASSED
> 
> **Failed Expectations Log:**
> * *None. All numerical ranges, schemas, and categorical sets passed successfully.*

---

### Dataset C: Diabetes BRFSS 2021

**Initial Exploration Metrics**
> **Rows:** 236,378    
> **Columns:** 22   
> **Duplicates:** 13,135     
> **Missing Values:** 0     
> 
> **Data Types:** > float64: 22 columns -> `['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']`

**Rule-Based Validation**
> **Overall Result:** FAILED
> 
> **Failed Expectations Log:**
> * **[FAIL]** Categorical Value Set Match (`expect_column_values_to_be_in_set`)
>   * **Column:** `Income`
>   * **Issues:**  66243 unexpected values found. ≈28.02% of 236378 total rows. 
>   * **Sample:**   
9.0     
10.0    
11.0

---