# attack framework
This is a framework for running attacks on sparsechem models.

[[_TOC_]]

## Setup
1. Enter the `data/` directory and extract all files.
   Find further instructions there.
2. Install requirements: `pip install -r requirements.txt`
3. Install local packages under the `packages/` directory.
  Find further instructions there.

## Parameters
### Required parameters
Some parameters that need to be set in order to run an attack.

* `--attack_name`: This specifies which attack to run. Values can be: `[NGMA, TrunkActivation, NOATTACK, Leaving, ActiveTrunkActivation, MMTrunkActivation]`.
`NOATTACK` is a special setting where we don't run any attacks.
  
* `--results_file`: Name of the .csv file you want to save your results in.
  This file contains the model config parameters, the attack config parameters and the attack's results (for example: some performance values like accuracy).
  If file already exists, new results will be appended. Example: `results.csv`
*Please note that the results from two different attacks **can not be saved in the same results file**.*
  *(This is because all attacks have different configs, and evaluation parameters.*

* `--seed`: Sets a seed for running the attack and training the model.

* `--model_save`: Whether or not the trained model should be saved. Possible values: `['yes', 'true', 't', 'y', '1']` and `['no', 'false', 'f', 'n', '0']`.
* `--model_save_path`: Specifies the path where the server and client models (+model config) should  be saved / should be loaded from. Default value is `None`.

#### Examples:
An example where the trained model is not saved for later atacks:

`python launch_attack.py --attack_name NGMA --results_file ngma_results.csv --model_save no --seed 42 --num_samples 100`

An example where we save the trained model for later attacks (in `trained_models/` subdirectory - this is created automatically):

`python launch_attack.py --attack_name NGMA --results_file ngma_results.csv --model_save yes --model_save_path trained_models/ --seed 42 --num_samples 100`

An example where we load the trained model for an attack (from `trained_models/`), and launch the attack with a different seed:

`python launch_attack.py --attack_name NGMA --results_file ngma_results.csv --model_save no --model_save_path trained_models/ --seed 89 --num_samples 100`

*Note*: When loading an already saved model, there is no use in setting ModelConfig parameters, such as `--hidden_sizes` or `--rounds` because they will be overwritten with the loaded model's.
### Model parameter default values
The default values for the model parameters have been set to match the Y1 run.
* **rounds**: 1000
* **hidden sizes**: [6000]
* **last dropout**: 0.2
* **weight decay**: 1e-6
* **last non linearity** and **non linearity**: relu
* **lr**: 1e-3
* **lr steps**: [10]

If you want to view the full list of parameters and their default values, please refer to `packages/attacks/attacks/configs.py`.

#### Evaluation
- average logloss
- average AUC-PR

Evaluations are calculated on the training set.
Averages are calculated over the clients.

## Attacks
### Trunk Activation Attack
The attacker model is a `RandomForestClassifier`, whose input is the trunk activation values.
The attacker's goal is to predict membership based on this activation.
#### Parameters
* `--num_samples`: Number of member and non-member samples that the attacker collects for training and evaluating her attack.
  Default is 500.
* `--n_estimators`: Number of estimators used to train the RandomForestModel. 
  Default is 100.
#### Evaluation
* TP, FP, TN, FN
* accuracy
* precision
* recall

### Active Trunk Activation Attack
This is the same attack as above, except that the adversary (the server) actively isolates participants.
He does not train one globally shared model, but multiple individual models, each belonging to one participant.
Attacking the individual models (instead of the global one) allows him to conduct a more successful membership inference attack.
The evaluation metrics are the averages of the attacks on the individual models.
#### Parameters
* `--num_samples`: Number of member and non-member samples that the attacker collects for training and evaluating her attack.
  Default is 500.
* `--n_estimators`: Number of estimators used to train the RandomForestModel. 
  Default is 100.
#### Evaluation
* TP, FP, TN, FN
* accuracy
* precision
* recall

### NGMA Attack
The attack exploits the input's unique fingerprint, which is reflected in the gradients as well.
It uses a majority voting scheme to determine whether a certain compound was used in a particular round.
#### Parameters
* `--num_samples`: Number of member and non-member samples that the attacker collects for training and evaluating her attack.
  It takes 30-50 seconds to evaluate one sample. Default value is 50.
* `--voting_threshold`: The fraction of gradients connected to a non-zero input that need to be non-zero. Affects attack precision. Usually a higher threshold means higher precision (higher confidence in positive predictions). Default value is 0.5.
#### Evaluation
* TP, FP, TN, FN
* accuracy
* precision
* recall

### Leaving / N-1 Attack
The attack simulates a setting where all participants train together for some rounds, then one of the participants leaves.
The attacker observes the success of an MI attack on a **specific** compound belonging to the "leaving party".
If the attacker success decreases after the participant leaves, he/she can be sure that the sample belonged to them.
This is performed with a binomial test with significance threshold of 5%.
#### Parameters
* `--num_epochs`: Number of epochs that the attacker tests his hypothesis on. 
  One epoch is equal to 50 rounds. It takes 20 minutes to evaluate one epoch. 
  Default is 30.
* `--voting_threshold`: The fraction of gradients connected to a non-zero input that need to be non-zero. Affects attack precision. Usually a higher threshold means higher precision (higher confidence in positive predictions). Default value is 0.2.
#### Evaluation
- number of positive epochs together
  - How many epochs had a positive membership inference prediction while all members trained *together*.
- number of positive epochs left
  - How many epochs had a positive membership inference prediction after one of the participants *left*.
- p
  - Probability of the two samples (`pos_epochs_together` and `pos_epochs_left`) coming from the same distribution.
  Any value over the 5% significance threshold means that there was no significant difference between them.


### Multi-Model Trunk Activation Attack
This is a Trunk Activation attack which is performed on the trunk outputs of multiple models simultaneously.
Each model (or CP - Compute Plan) is distinct. 
The learning rate for each CP is randomly generated (a number which is at most 0.1).
By default the adversary attacks the model from the last epoch of each CP.

#### Parameters
* `--num_models`: Number of compute plans to train for an attack.
* `--num_samples`: Number of member and non-member samples that the attacker collects for training and evaluating her attack. Default is 500.
* `--attack_type`: Type of attacker model. Possible values: `nn` - neural network, `rf` - random forest or `gb` - gradient boosting. Default value is 'rf'.
* `--num_epochs`: Number of epochs that the attacker trains for (if attacker type is `nn`).
* `--n_estimators`: Number of estimators used to train the RandomForestModel. Default is 100.
* `--best_models`: Whether to use 'best' model additionally as attack input. Possible values: `['yes', 'true', 't', 'y', '1']` and `['no', 'false', 'f', 'n', '0']`.
* `--intermediate_models`: Whether to use intermediate models of 'best' model additionally as attack input. Possible values: `['yes', 'true', 't', 'y', '1']` and `['no', 'false', 'f', 'n', '0']`.
* `--attacked_epochs`: Sequence of attacked epochs for each compute plan. If set, this option overrides the default attack, where only the last epoch is attacked. Default value is `None`.

#### Evaluation
* TP, FP, TN, FN
* accuracy
* precision
* recall
