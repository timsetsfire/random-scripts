#!/usr/bin/env python
# coding: utf-8

#import egg files for libraries
import pandas as pd 
import io
import sys
# python imports
import json
from datetime import datetime
import os
import argparse
import yaml
import logging
from easydict import EasyDict
import os
import datarobot as dr

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

class ConfigKeys:
    TOKEN = "token"
    ENDPOINT = "endpoint"
    MODEL_ID = "model_id"
    PROJECT_ID = "project_id"
    USECASE_ID = "usecase_id"
    DATASET_ID = "dataset_id"
    DATASET_VERSION_ID = "dataset_version_id"
    ALL = [TOKEN, ENDPOINT, MODEL_ID, PROJECT_ID, USECASE_ID, DATASET_ID, DATASET_VERSION_ID]
    DST = [TOKEN, ENDPOINT]

def validate_src_dst_config(src_yaml, dst_yaml):
    with open(src_yaml, "r") as f:
        src_dict = yaml.load(f, Loader = yaml.SafeLoader)
    with open(dst_yaml, "r") as f:
        dst_dict = yaml.load(f, Loader = yaml.SafeLoader)
    src_keys = list(src_dict.keys())
    dst_keys = list(dst_dict.keys())
    for k in ConfigKeys.ALL:
        if k not in src_keys:
            raise Exception(f"src yaml expected to have key {k}, but found {src_keys}")  
    for k in ConfigKeys.DST:
        if k not in dst_keys:
            raise Exception(f"dst yaml expected to have key {k}, but found {dst_keys}")
    if src_dict["dataset_id"] is None:
        raise Exception("dataset id must be set in src config")                 
    return EasyDict(src_dict), EasyDict(dst_dict)   

parser = argparse.ArgumentParser() 
parser.add_argument("--src-dr-yaml", 
                    type=str, 
                    default = "saas_config.yaml", 
                    help="path to yaml containing endpoint url and token to access source datarobot instance")
parser.add_argument("--dst-dr-yaml", 
                    type=str, 
                    default = "saas_config.yaml", 
                    help="path to yaml containing endpoint url and token to access destination datarobot instance")
parser.add_argument("--logging-level", type=str, default="DEBUG")


# # set relative path for capturing pertinent information
current_dir = os.getcwd()
## set time of execution
t = datetime.now().strftime('%Y%m%d-%H%M%S')
## set name of logfile to include MRM and time of promotion
logfile_name = f'migration_{t}.log'

# args = {"src_dr_yaml": "dep0_config.yaml", "dst_dr_yaml": "dep1_config.yaml", "logging_level": "DEBUG"}
# args = EasyDict(args)
args = parser.parse_args()

## set up logger
logging.basicConfig(
        format="{} - %(levelname)s - %(asctime)s - %(message)s".format(__name__),
        handlers=[
            logging.FileHandler(logfile_name),
            logging.StreamHandler()
        ]
)
logger = logging.getLogger(__name__)
logger.setLevel(args.logging_level)
        

logger.info("validating src and dst config yaml files")

## validate config yaml files
src_dict, dst_dict = validate_src_dst_config(args.src_dr_yaml, args.dst_dr_yaml)        

# datarobot imports and assertions

client = dr.Client(token=src_dict.token, endpoint=src_dict.endpoint)
# # assert(dr.__version__ == "2.19.0")

# # supress ssl warnings
# import warnings
# warnings.filterwarnings('ignore')

# # initialize python dict to capture model and project information
model_migration_identification = {}

logger.info('Collecting information from model selected for migration')
project_id = src_dict.project_id
model_id = src_dict.model_id
dataset_id = src_dict.dataset_id
dataset_version_id = src_dict.dataset_version_id
usecase_id = src_dict.usecase_id

usecase = dr.UseCase.get(usecase_id)

# grab dataset from catalog 
dataset = dr.Dataset.get(dataset_id)
logger.info(f"Retrieving datasets {dataset.name} from Dataset Registry")
data_path = os.path.join(script_directory, dataset.name)
if src_dict.dataset_version_id is None:
    dataset.get_file(file_path = data_path)
else:
    data = client.get(f"datasets/{dataset_id}/versions/{dataset_version_id}/file")
    pd.read_csv(io.BytesIO(data.content)).to_csv(data_path)

logger.info(f"datasets {dataset.name} is available locally at {data_path}")
# # set name of model promotion json object
model_migration_filename = f'model_migration_{t}.json'

# make a directory to capture model_promotion_logs
os.makedirs('model_promotion_logs', exist_ok=True)

# create and open log file to capture pertinent information
logfile = open(
        os.path.join(current_dir, 'model_promotion_logs', logfile_name), 'w+'
        )

# set current time to log how long sections of model promotion are taking
ct = datetime.now().strftime('%Y%m%d-%H%M%S')

# write header and dr api version to logfile
logger.info(f'Promotion Start time = {ct}')
logger.info(f'DataRobot Python API: {dr.__version__}')
logger.info(f'{src_dict.endpoint.replace("/api/v2", "")} Model ID: {model_id}')
logger.info(f'{src_dict.endpoint.replace("/api/v2", "")} Dataset ID: {dataset_id} Dataset Version ID: {dataset_version_id}')

# # lower level DataRobot client connection
src_client = dr.Client(token = src_dict.token, endpoint = src_dict.endpoint)

# # set project to selected project
selected_project = dr.Project.get(project_id)
# get advanced options
selected_project_adv_options = selected_project.advanced_options.__dict__
# if project_info.accuracy_optimized:
#     selected_project_adv_options['accuracy_optimized_mb'] = True
    
# # remove scoring code only option as scoring code is not enabled
# try:
#     del selected_project_adv_options['scoring_code_only']
# except KeyError:
#     logger.info('scoring code only not applicable for this model and project')

# # get selected model
selected_model = dr.Model.get(project=project_id, model_id=model_id)

# # Get Blueprint of the selected model
selected_bp = dr.Blueprint.get(project_id, selected_model.blueprint_id)

# # set current time to log how long sections of model promotion are taking
ct = datetime.now().strftime('%Y%m%d-%H%M%S')

# write information to logfile
logger.info(f'Selected Model Information:')
logger.info(f'-----------------------------------------')
logger.info(f'Time Check = {ct}')
logger.info(f'Selected model blueprint: {selected_bp.model_type}')
logger.info(f'Selected model blueprint processes: '\
              f'{json.dumps(selected_bp.processes)}')

# get features from selected model
# selected_features_raw = selected_model.get_features_used()
selected_features = dr.Featurelist.get(
    selected_project.id, selected_model.featurelist_id
).features

# # get model parameters
selected_params = selected_model.get_advanced_tuning_parameters()

# # get sample percentage and nrows of training data
selected_sample_pct = selected_model.sample_pct
selected_model_nrows = selected_model.training_row_count

# # get optimization metric
metric = selected_project.metric

# # get target
target = selected_project.target

# # get positive class of target if classification
if selected_project.target_type == 'Binary':
    positive_class = selected_project.positive_class

# # write information to logfile
logger.info(f'Selected model features: {json.dumps(selected_features)}')
logger.info(f'Selected model parameters: {json.dumps(selected_params)}')
logger.info(f'Selected model sample percentage: '\
              f'{json.dumps(selected_sample_pct)}')
logger.info(f'Selected optimization metric: {metric}')
logger.info(f'Selected target: {target}')

#if it is frozen than we need to know the parent to retrace from there
if selected_model.is_frozen:
    logger.info("Frozen Model Run: Switch to Parent")
    frozen_model = dr.FrozenModel.get(project_id, model_id)
    selected_parent_model = dr.Model.get(
            project=project_id, model_id=frozen_model.parent_model_id
            )
    parent_sample_pct = selected_parent_model.sample_pct
    parent_model_nrows = selected_parent_model.training_row_count
    selected_parent_params = selected_parent_model.get_advanced_tuning_parameters()
    selected_bp = dr.Blueprint.get(selected_project.id, selected_parent_model.blueprint_id)
    logger.info(selected_parent_model.id)

# unlock holdout and confirm holdout performance
selected_project.unlock_holdout()

# # resynch to model after unlocking holdout
selected_model = dr.Model.get(project=project_id, model_id=model_id)

validation_performance = selected_model.metrics[metric]['validation']
holdout_performance = selected_model.metrics[metric]['holdout']
cv_score = selected_model.get_cross_validation_scores()['cvScores'][metric]

# # write information to logfile
logger.info(f'Out of sample validation performance: ' \
              f'{validation_performance}')
logger.info(f'Cross-Validation performance: {json.dumps(cv_score)}')
logger.info(f'Holdout sample performance: {holdout_performance}')

# # Calculate predictions on holdout partition of dataset
try:
    selected_holdout_predictions_job = (
            selected_model
            .request_training_predictions(dr.enums.DATA_SUBSET.HOLDOUT)
            )
    selected_holdout_predictions = (
            selected_holdout_predictions_job.get_result_when_complete()
            )
    
    # Fetch holdout predictions as data frame
    df_selected_holdout_predictions = (
            selected_holdout_predictions.get_all_as_dataframe()
            )
except:
    # get holdout predictions if already calculated
    all_holdout_predictions = [
        p for p in dr.TrainingPredictions.list(project_id=selected_project.id) 
        if p.data_subset == 'holdout'
    ]
    
    # match prediction model id to get correct set of holdout predictions
    selected_holdout_predictions = (
            [p for p in all_holdout_predictions 
             if p.model_id == selected_model.id][0]
            )
   
    # Fetch holdout predictions as data frame
    df_selected_holdout_predictions = (
            selected_holdout_predictions.get_all_as_dataframe()
            )

# collect pertinent information from origination project and model
owner = (
        [user.username for user in selected_project.get_access_list() 
        if user.role == 'OWNER'][0]
        )
model_migration_identification[dst_dict.endpoint.replace("/api/v2", "")] = {
    'owner': owner,
    'project_id': selected_project.id,
    'model_id': selected_model.id,
    'frozen_model': selected_model.is_frozen,
    'created_on': selected_project.created.strftime(
            format="%m/%d/%Y, %H:%M:%S"),
    'project_name': selected_project.project_name,
}

logger.info(f'Migrating model from {src_dict.endpoint.replace("/api/v2", "")} '\
      f'to {dst_dict.endpoint.replace("/api/v2", "")}')

# Higher Level DR environment client connection
dst_client = dr.Client(token = dst_dict.token, endpoint = dst_dict.endpoint)

dst_usecase = dr.UseCase.create(name = usecase.name, description = usecase.description)


logger.info('beginning migration of selected model')

# set time of execution
t = datetime.now().strftime('%Y%m%d-%H%M%S')



# set advanced options
if not selected_project_adv_options['response_cap']:
    
    selected_project_adv_options['response_cap'] = 1.0

# #set advanced options for migrated project
logger.info('setting advanced options')    
advanced_options = dr.AdvancedOptions(**selected_project_adv_options)

# create a project in new environment

dst_dataset = dr.Dataset.create_from_file(data_path) 
dst_usecase.add(dst_dataset)

try:
    logger.info("creating project")
    migrated_project = dr.Project.create_from_dataset(
        dataset_id=dst_dataset.id,
        project_name=f'migration_{selected_project.project_name}_{t}', 
        use_case = dst_usecase)
except Exception as e:
    logger.error(f"data_path = {data_path}")
    logger.error(e)

# set partitioning scheme
logger.info(f"selected project partitioning scheme: {selected_project.partition}")
if selected_project.partition['cv_method'] == 'stratified':
    migrated_project_partition = dr.StratifiedCV(
        holdout_pct=selected_project.partition['holdout_pct'], 
        reps=selected_project.partition['reps'], 
        seed=1234
    )
elif selected_project.partition["cv_method"] == 'random':
    migrated_project_partition = dr.RandomCV(
        holdout_pct=selected_project.partition['holdout_pct'], 
        reps=selected_project.partition['reps'], 
        seed=1234
    )

# set target, metric and method
if selected_project.target_type == 'Binary':
    migrated_project.analyze_and_model(
        target=target,
        positive_class=positive_class,
        metric=metric,
        advanced_options=advanced_options,
        partitioning_method=migrated_project_partition,
        mode="manual")
else:
    migrated_project.analyze_and_model(
        target=target,
        metric=metric,
        advanced_options=advanced_options,
        partitioning_method=migrated_project_partition,
        mode="manual")
    

# create feature list from selected features
migrated_featurelist = (
        migrated_project
        .create_featurelist('selected_model_features', selected_features)
        )

# get list of all potential blueprints in the migrated_project repository
migrated_blueprints = migrated_project.get_blueprints()

# test for match to selected blueprint
bp_matches = []
for bp in migrated_blueprints:
    if sorted(bp.processes) == sorted(selected_bp.processes):
        bp_matches.append(bp)
logger.info(f'number of matching blueprints = {len(bp_matches)}')



# check whether a frozen model was selected
if selected_model.is_frozen == True:
    # unlock holdout to build on more training data
    migrated_project.unlock_holdout()
    sample_pct = parent_sample_pct
    training_row_count = parent_model_nrows
    logger.info('selected model is a frozen model')
else:
    sample_pct = selected_sample_pct
    training_row_count = selected_model_nrows

# train models using matching blueprint, sample percentage and feature list    
for bp in bp_matches:
    logger.info(bp.id)
    migrated_model_job_id = migrated_project.train(
        bp.id, 
#         sample_pct=sample_pct, 
        training_row_count = training_row_count,
        featurelist_id=migrated_featurelist.id,
    )

    # wait for model training to complete
    migrated_model = dr.models.modeljob.wait_for_async_model_creation(
        project_id=migrated_project.id,
        model_job_id=migrated_model_job_id,
        max_wait=86400
    )

    # check for frozen model
    # and train a frozen model derived from the parent if frozen
    if selected_model.is_frozen == True:
        # first check if migrated parent model has out of sample performance that matches selected model parent
        if (migrated_model.metrics[metric]['validation'] == 
            selected_parent_model.metrics[metric]['validation']):
            logger.info(f'Validation scores match for parent model {bp.id}')
            # run cross validation on migrated project
#             migrated_model.cross_validate()
        else:
            migrated_model.delete()
            logger.info('selected parent', selected_parent_model.metrics[metric]['validation'])
            logger.info('migrated parent', migrated_model.metrics[metric]['validation'])
            logger.info('non-matching parent model detected and deleted')
            continue
        
        JobId = (
                migrated_model.request_frozen_model(
                    training_row_count=selected_model_nrows
#                     sample_pct=selected_sample_pct
                )
        )
        migrated_model = dr.models.modeljob.wait_for_async_model_creation(
            project_id=migrated_project.id, 
            model_job_id=JobId.id,
            max_wait=86400
        )
        
        # check if migrated_model has out of sample performance 
        # that matches selected model
        if (migrated_model.metrics[metric]['validation'] == 
            selected_model.metrics[metric]['validation']):
            logger.info(f'Validation scores match for {bp.id}')
            # run cross validation on migrated project
            try:
                migrated_model.cross_validate()
            except Exception as e:
                logger.info(e)
            break
        else:
            migrated_model.delete()
            logger.info('selected ', selected_model.metrics[metric]['validation'])
            logger.info('migrated ', migrated_model.metrics[metric]['validation'])
            logger.info('non-matching model detected and deleted')
            try:
                migrated_parent_model = (
                    dr.Model.get(
                            project=migrated_project.id, 
                            model_id=frozen_model.selected_parent_model_id
                            )
                )
            except:
                logger.info('non-matching models detected')
            
    else:
        # check if migrated_model has out of sample performance
        # that matches selected model
        if (migrated_model.metrics[metric]['validation'] == 
            selected_model.metrics[metric]['validation']):
            logger.info(f'Validation scores match for {bp.id}')
            # run cross validation on migrated project
            try:
                migrated_model.cross_validate()
            except Exception as e:
                logger.info(e)
            break
        else:
            migrated_model.delete()
            logger.info('selected ', selected_model.metrics[metric]['validation'])
            logger.info('migrated ', migrated_model.metrics[metric]['validation'])
            logger.info('non-matching models detected and deleted')


### Verify Performance of Migrated Model on out of Sample Data

# unlock holdout
migrated_project.unlock_holdout()

# print list of migrated models
logger.info(migrated_project.get_models())

# verify out of sample performance
migrated_model = [m for m in migrated_project.get_models() 
                  if m.model_type == selected_model.model_type 
                  and m.sample_pct == selected_model.sample_pct][0]

# get out of sample performance performance information
migrated_validation_score = migrated_model.metrics[metric]['validation']
migrated_cv_scores = (migrated_model
                      .get_cross_validation_scores()['cvScores'][metric]
                      )
migrated_holdout_score = migrated_model.metrics[metric]['holdout']

# set current time to log how long sections of model promotion are taking
ct = datetime.now().strftime('%Y%m%d-%H%M%S')


# write information to logfile
logger.info('Migrated Model Information')
logger.info('--------------------------------------------')
logger.info(f'Time Check = {ct}')
logger.info(f'{dst_dict.endpoint.replace("/api/v2", "")} Project ID: {migrated_project.id}')
logger.info(f'{dst_dict.endpoint.replace("/api/v2", "")} Model ID: {migrated_model.id}')
logger.info(f'Out of sample validation performance: '\
              f'{migrated_validation_score}')
logger.info(f'Cross-Validation performance: '\
              f'{json.dumps(migrated_cv_scores)}')
logger.info(f'Holdout sample performance: {migrated_holdout_score}')

### Confirm model parameters / hyperparameters match selected model

# get features from development model
migrated_features = migrated_model.get_features_used()

# get model parameters
migrated_params = migrated_model.get_advanced_tuning_parameters()

# get sample percentage
migrated_sample_pct = migrated_model.sample_pct

# get optimization metric
migrated_metric = migrated_project.get(migrated_project.id).metric

# get target
migrated_target = migrated_project.get(migrated_project.id).target

# set current time to log how long sections of model promotion are taking
ct = datetime.now().strftime('%Y%m%d-%H%M%S')

# write information to logfile
logger.info(f'Time Check = {ct}')
logger.info(f'migrated model features: {json.dumps(migrated_features)}')
logger.info(f'migrated model parameters: {json.dumps(migrated_params)}')
logger.info(f'migrated model sample percentage: '\
              f'{json.dumps(migrated_sample_pct)}')
logger.info(f'migrated optimization metric: {metric}')
logger.info(f'migrated target: {target}')

# verify matching parameters / hyperparameters
# set current time to log how long sections of model promotion are taking
ct = datetime.now().strftime('%Y%m%d-%H%M%S')

logger.info(f'Time Check = {ct}')
logger.info('Migrated Replication Checks')
logger.info('--------------------------------------------')

# verify that out of sample perfomance is identical
if migrated_model.metrics[metric] == selected_model.metrics[metric]:
    logger.info('Has out of sample performance been confirmed?: True')
    logger.info('Migrated model out of sample performance verification: '
                  'True')
else:
    logger.info('Has out of sample performance been confirmed?: False')
    logger.info('Migrated model out of sample performance verification: '\
                  'False')

logger.info(f'Migrated Features Match: '\
              f'{ len(set(migrated_features).intersection(set(selected_features))) / len(set(migrated_features).union(set(selected_features))) == 1.0}')
logger.info(f'Migrated Parameters Match: '\
              f'{migrated_params == selected_params}')
logger.info(f'Migrated Sample Percentage Match: '\
              f'{migrated_sample_pct == selected_sample_pct}')
logger.info(f'Migrated Optimization Metric Match: '\
              f'{migrated_metric == metric}')
logger.info(f'Migrated Target Match: {migrated_target == target}')

### Validate Matching Predictions on Holdout or Test Data

# Calculate predictions on holdout partition of dataset
try:
    migrated_holdout_predictions_job = (
            migrated_model
            .request_training_predictions(dr.enums.DATA_SUBSET.HOLDOUT)
            )
    migrated_holdout_predictions = (
            migrated_holdout_predictions_job.get_result_when_complete()
            )

#     Fetch hodout predictions as data frame
    df_migrated_holdout_predictions = (
            migrated_holdout_predictions.get_all_as_dataframe()
            )
except:
    # get holdout predictions if already calculated
    migrated_holdout_predictions = [
        p for p in dr.TrainingPredictions.list(project_id=migrated_project.id) 
        if p.data_subset == 'holdout'
    ][0]
    
#     Fetch holdout predictions as data frame
    df_migrated_holdout_predictions = (
            migrated_holdout_predictions.get_all_as_dataframe()
            )

# validate that the holdout predictions are identical
if df_selected_holdout_predictions.equals(df_migrated_holdout_predictions):
    logger.info('Are the holdout predictions of the migrated model '\
          'identical to the selected model?: True')
    logger.info('Migrated model holdout predictions validated: True')
else:
    logger.info('Are the holdout predictions of the migrated model '
          'identical to the selected model?: False')
    logger.info('Migrated model holdout predictions validated: False')

# save samples of holdout predictions for verification by MRM
df_selected_holdout_predictions.sample(
        n=1000, random_state=1234,replace=True).to_csv(
                'selected_holdout_predictions_sample.csv', index=False)
df_migrated_holdout_predictions.sample(
        n=1000, random_state=1234,replace=True).to_csv(
                'migrated_holdout_predictions_sample.csv', index=False)

# close out logfile
# logfile.close()

# collect pertinent information for destination
model_migration_identification[dst_dict.endpoint.replace("/api/v2", "")] = {
    'owner': owner,
    'project_id': migrated_project.id,
    'model_id': migrated_model.id,
    'frozen_model': migrated_model.is_frozen,
    'created_on': migrated_project.created.strftime(
            format="%m/%d/%Y, %H:%M:%S"),
    'project_name': migrated_project.project_name,
}

# add migration id to model promotion identification
model_migration_identification['migration_information'] = {
        'migration_ID': (f'{selected_project.id}:{selected_model.id}-'\
                         f'{migrated_project.id}:{migrated_model.id}'),
        'promoter': migrated_project.get_access_list()[0].username,
        'replicated': df_selected_holdout_predictions.equals(
                df_migrated_holdout_predictions)
        }

logger.info("Registering migrated model in destination DataRobot environment")
registered_model = dr.RegisteredModelVersion.create_for_leaderboard_item(model_id = migrated_model.id)
dst_dict["registered_model_id"] = registered_model.registered_model_id
dst_dict["registered_model_version_id"] = registered_model.id
dst_dict["usecase_id"] = dst_usecase.id
dst_dict["project_id"] = migrated_project.id
dst_dict["model_id"] = migrated_model.id
logger.info(f"Registeration is complete.  you can review the details at {args.dst_dr_yaml}")

with open(args.dst_dr_yaml, 'w') as f:
    yaml.dump(dict(dst_dict), f)

# with open("model_promption_info.json", 'w') as f:
#     json.dump(model_migration_identification, f)
