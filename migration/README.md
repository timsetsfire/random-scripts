move your model from one instance of DR to another.  

# required
One config yaml concerning source. This will look like 

```
usecase_id: 683a0175a85e88ad2853dc8b
project_id: 683a028222d1aa17f0e20927
model_id: 683a076ff9bf95a27f7e31b4
registered_model_id: 688b9a767655fcd6a45b1a12
registered_model_version_id: 688b9a787655fcd6a45b1a1a
dataset_id: 68387f74f7c27991354c65d9
dataset_version_id: null
endpoint: https://app.datarobot.com/api/v2
token: your-datarobot-api-token
```
and one yaml for the destination 
```
endpoint: https://app.datarobot.com/api/v2
token: your-datarobot-api-token
```

# Usage 

`python model_migration_v3.py --src-dr-yaml ./src.yaml --dst-dr-yaml ./dst.yaml`

This does not currently support Time Series Model, and might not have complete coverage.  

# What this does 

This will copy the dataset from src instance of datarobot to the dst instance. 
it will then create a model identical to the original model
it will then register that model in the dst instance of datarobot.  
