# Situated Interactive MultiModal Conversations (SIMMC) Challenge 2020

## Model Instructions
base directory: simmc/mm_action_prediction/

### (1) Generate predicted belief state

```
$ scripts/
```

### (2) Preprocess Dataset

preprocess dataset with predicted belief state
```
$ scripts/test_belief_preprocess_simmc.sh
```

### (3) Train the model
```
$ scripts/test_belief_simmc_model.sh
$ scripts/test_belief_simmc_all_model.sh
```

### (4) Inference the model
```
$ scripts/test_belief_inference_simmc_model.sh
```

### (5) Ensemble models and Generate submission files
ex) --model_types model1 model2 model3 ...

--best_gen model1
    
--model_type inputs are models that used for action prediction and retrieval prediction.

--best_gen inputs is a model that used for response generation


```
$ python ensemble_code/furniture_ensemble.py --model_types HAE_R300 HRE_R300_MAG --best_gen HAE_R300
$ python ensemble_code/fashion_ensemble.py --model_types T_HAE_G300_TD MN_R300_MAG_TD --best_gen MN_G300_TD
```

