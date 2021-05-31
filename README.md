# Capstone Project - Heart Failure Prediction

These are the results of the final project for Udacity's Machine Learning Engineer for Microsoft Azure Nanodegree.

The project's goal was to create two models in Azure using AutoML and Hyperdrive, to deploy the best performing model and to interact with the model via web services.

The data used to train the model was required to be external and not available in the Azure's ecosystem.


## Project Set Up and Installation

The workflow of the project is illustrated in the following diagram.

![Project Workflow](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/capstone-diagram.png)


## Dataset

### Overview

The data analyzed during the project is the "Heart failure clinical records Data Set" from UCI Machine Learning Repository. ( Source: [Heart failure Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) )  

This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

Column | Description
------ | -----------
age | Age of the patient (years)
anaemia | Decrease of red blood cells or hemoglobin (boolean)
high blood pressure | If the patient has hypertension (boolean)
creatinine phosphokinase (CPK) | Level of the CPK enzyme in the blood (mcg/L)
diabetes| If the patient has diabetes (boolean)
ejection fraction | Percentage of blood leaving the heart at each contraction (percentage)
platelets | Platelets in the blood (kiloplatelets/mL)
sex | Woman or man (binary)
serum creatinine | Level of serum creatinine in the blood (mg/dL)
serum sodium | Level of serum sodium in the blood (mEq/L)
smoking | If the patient smokes or not (boolean)
time | Follow-up period (days)
[target] death event | If the patient deceased during the follow-up period (boolean)


### Task

Based on the 12 features of the medical records a model is built to predict mortality by heart failure, mapped onto the target column ("death event"). This is a binary classification problem, where the outcome is either 1 (death) or 0 (no death).


### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
