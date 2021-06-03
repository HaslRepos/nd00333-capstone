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

Based on the 12 features of the medical records a model is built to predict mortality by heart failure. This is a binary classification problem, where the outcome is either 1 (death) or 0 (no death) provided in the target column ("death event").


### Access

The dataset is available on UCI: [Heart failure Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

The file is downloaded from the url, registred in the workspace and converted to a dataframe.

```python
example_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'
dataset = Dataset.Tabular.from_delimited_files(example_data)        

# Register Dataset in Workspace
dataset = dataset.register(workspace=ws, name=key, description=description_text)

# Convert dataset to dataframe
df = dataset.to_pandas_dataframe()
```


## Automated ML

AutoML offers a broad range of parameters to customize the experiments. The following parameters are chosen for Heart Failure Prediction project:

```python
automl_settings = {
    "experiment_timeout_minutes": 25,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'accuracy'
}

automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT",   
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )

```


* *experiment_timeout_minutes*: The experiment terminates after 25 minutes, which gives it enough time for the small dataset.
* *max_concurrent_iterations*: 4 concurrent iterations are allowed. AmlCompute clusters support one interation running per node. The sum of the max_concurrent_iterations values for all experiments should be less than or equal to the maximum number of nodes.
* *primary_metric*: The metric that Automated Machine Learning will optimize for model selection. Accuracy is the standard for classification tasks.
* *task*: The type of task to run depending on the type of automated ML problem to solve. The heart failure prediction is a classification problem.
* *label_column_name*: The name of the label column.
* *featurization*: Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used.


### Results

The best performing model is the **VotingEnsemble** with an accuracy of **0.8730**.

A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble. (Source: [How to Develop Voting Ensembles With Python](https://machinelearningmastery.com/voting-ensembles-with-python/))


**Run Details**

![AutoML Run Details](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/automl_run_details.png)


**Best Model**

![AutoML Best Model 1](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/automl_best_model_1.png)

![AutoML Best Model 2](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/automl_best_model_3.png)


Further progress could be made by evaluating other metrics instead of Accuracy, such as AUC. AutoML also offers much more possibilities for configuration than the ones used in this project. Additional improvement might be achieved by exploring these possibilities.


## Hyperparameter Tuning

Hyperparameter tuning, also called hyperparameter optimization, is the process of finding the configuration of hyperparameters that results in the best performance. Azure Machine Learning lets you automate hyperparameter tuning and run experiments in parallel to efficiently optimize hyperparameters. (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
The Heart Failure Prediction experiment is based on a Logistic Regression algorithm.

```python
param_sampling = RandomParameterSampling(
                    {
                        '--C': choice(0.01, 0.1, 1.0, 10.0, 100.0),
                        '--max_iter': choice(25, 50, 75, 100, 125, 150)
                    }
                 )

estimator = SKLearn(source_directory = os.path.join("./"),
                entry_script='train.py',
                compute_target=compute_target
            )

hyperdrive_run_config = HyperDriveConfig(hyperparameter_sampling=param_sampling,
                            primary_metric_name='Accuracy',
                            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                            estimator=estimator,
                            policy=early_termination_policy,
                            max_total_runs=20,
                            max_concurrent_runs=4
)
```

In random sampling, hyperparameter values are randomly selected from the defined search space.

* *--C* (float): Inverse of regularization strength
* *max_iter* (int): Maximum number of iterations taken for the solvers to converge


### Results

The best performing logistic regression model has an accuracy of **0.783**, achieved with the following parameters:

Hyperparameter | Value
-------------- | -----
Regularization Strength | 1.0
Max Iterations | 100


**Run Details**

![HyperDrive Run Details 1](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/hyperd_run_details_1.png)

![HyperDrive Run Details 2](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/hyperd_run_details_2.png)


**Best Model**

![HyperDrive Best Model](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/hyperd_best_model_3.png)


Further improvement of the results might be made by choosing a different metric to be optimized (such as AUC) or by selecting another algorithm (like xgboost, etc). One could also replace Random Sampling by Bayesian Sampling, which selects the hyperparameters based on previous performance.


## Model Deployment

The AutoML model (voting ensemble) has been chosen to be deployed since it outperforms the HyperDrive model.

Deploying a model in Azure requires several steps:

### Register the best model

```python
best_run.download_file('/outputs/model.pkl', os.path.join('./model','automl-heart-failure-best-model'))
model = best_run.register_model(model_name = 'automl-heart-failure-best-model', model_path = './outputs/model.pkl')
```

### Prepare inference configuration

```python
best_run.download_file('outputs/conda_env_v_1_0_0.yml', 'conda_env.yml')
environment = Environment.from_conda_specification(name = 'heart-failure-environment', file_path = 'conda_env.yml')

best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'score.py')

inference_config = InferenceConfig(entry_script = 'score.py', environment = environment)

aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1, auth_enabled = True, enable_app_insights = True)
```

### Deploy the best model

```python
webservice = Model.deploy(workspace = ws, name = 'heart-failure-service', models=[model], inference_config = inference_config, deployment_config = aci_config, overwrite = True)

webservice.wait_for_deployment(show_output = True)
```

**Model Deployment**

![Deploy Best Model](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/Deploy_best_model.png)


Interaction with the model is possible as soon as the service is up and running.

### Prepare test data

```python
data = {"data": [{"age": 72.0,
                   "anaemia": 1,
                   "creatinine_phosphokinase": 110,
                   "diabetes": 0, "ejection_fraction": 25,
                   "high_blood_pressure": 0,
                   "platelets": 274000.0,
                   "serum_creatinine": 1.0,
                   "serum_sodium": 140,
                   "sex": 1,
                   "smoking": 1,
                   "time": 65},
                  {"age": 56.0,
                   "anaemia": 1,
                   "creatinine_phosphokinase": 135,
                   "diabetes": 1,
                   "ejection_fraction": 38,
                   "high_blood_pressure": 0,
                   "platelets": 133000.0,
                   "serum_creatinine": 1.7,
                   "serum_sodium": 140,
                   "sex": 1,
                   "smoking": 0,
                   "time": 244},
                 ]}
request = json.dumps(data)
```

### Send the request to the deployed webservice

```python
output = webservice.run(request)
output
```

```python
'{"result": [1, 0]}'
```

**Consume Best Model**

![Consume Best Model](https://github.com/HaslRepos/nd00333-capstone/blob/master/images/Consume_best_model.png)


## Screen Recording

 Screen Recording of the project is available on Google Drive:

[Capstone Project Screencast](https://drive.google.com/file/d/1Awq6KFDjnccQuJ5uSt7TWBoSssKyajQy/view?usp=sharing)


## Standout Suggestions

The project didn't focus on performance of the model trained and deployed. Within a production environment performance is key for any application.
Azure's Application Insights offers great possibilities to explore applications and diagnose problems.

Application Insights can be enabled before or after the deployment and probides a visualization of error rates or response tinmes.
