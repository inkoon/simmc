# Situated Interactive MultiModal Conversations (SIMMC) Challenge 2020

## Model Instructions
task1, task2 base directory: simmc/mm_action_prediction/

#### Brief Summary
We have built an end-to-end encoder-decoder model for sub-task1 & sub-task2.

Encoder is composed of three components: Utterance & History Encoder, Multi-Modal Fusion Gate, Belief State Embedder. Using HAE, HRE, MN Encoder from baseline, and two different types of embeddings (random, glove). Adding two more multi-modal fusion gate MAG (Wasifur Rahman et al., 2019) and MMI-module (Jianfei Yu et al., 2020). Further, our model use predicted belief-state from sub-task3.

We trained various types of models and generated final results by ensemble.

## (1) Generate predicted belief state
### Installation
Install the required Python packages:
- Python 3.6+
- Pytorch 1.5+
- Transformers 2.8.0 (IMPORTANT)
### Running model for Task3 and Task2

1. **Preprocess** the datasets to reformat the data for GPT-2 input.

```
$ cd mm_dst
$ ./run_preprocess_gpt2.sh
```
2. **Train** the baseline model

```
$ ./run_train_gpt2.sh [KEYWORD] [GPU_ID]
```

The shell script above repeats the following for both {furniture|fashion} domains.


3. **Generate** ensembled prediction for `devtest|test` data

```
pip uninstall transformers
pip install transformers -t transformers
mv transformers transformers_package
mv transformers_package/transformers transformers
cp modeling_utils.py

$ ./run_generate_using_ensemble.sh [GPU_ID]
```

The generation results are saved in the `/mm_dst/results` folder. Change the `path_output` to a desired path accordingly.


4. **Postprocess** predictions for `devtest|test` data

```
$ ./run_postprocess_gpt2.sh
```

Done! You can now evaluate Task3 and Task2 With generated files in the following directory
```
$ simmc/mm_dst/results/furniture/ensemble/
$ simmc/mm_dst/results/fashion/ensemble/
```

## (2) Preprocess Dataset
### Installation
Install the required Python packages:
- Python 3.6+
- Pytorch 1.5+
- Transformers 3.1.0 (IMPORTANT)


we use transformers 2.8.0 for generating belief state, but we use transformers 3.1.0 for training & inferencing task1 & task2


preprocess dataset with predicted belief state
```
$ ./scripts/belief_preprocess_simmc.sh
```

## (3) Train the model
```
$ ./scripts/belief_simmc_model.sh
$ ./scripts/train_simmc_model.sh
$ ./scripts/belief_simmc_all_model.sh
$ ./scripts/train_all_simmc_model.sh
```

## (4) Inference the model
```
$ ./scripts/belief_inference_simmc_model.sh
$ ./scripts/inference_simmc_model.sh
```

## (5) Ensemble models and Generate submission files
ex) --model_types model1 model2 model3 ...

--best_gen model1

--ret_model_types model1 model2 model3 ...
    
--model_types and --ret_model_types inputs are models that used for action prediction and retrieval prediction.

--best_gen inputs is a model that used for response generation


```
$ python ensemble_code/furniture_ensemble.py --model_types HRE_R300_S_TD HRE_R300_MAG_S_TD MN_G300_MAG_S HAE_G300_MAG_H768_S HAE_G300_MMI_H768_S --best_gen MN_R300_MAG_S_TD --ret_model_types MN_R300_gpt2S_TD HAE_R300_MAG_gpt2S HRE_R300_MAG_S_TD MN_G300_MAG_H768_S HAE_G300_MMI_H768_S
$ python ensemble_code/fashion_ensemble.py --model_types HRE_R300_S_TD HAE_R300_MAG_gpt2S_TD MN_G300_MAG_S HAE_G300_MAG_H768_S HAE_G300_MMI_H768_S --best_gen MN_R300_S_TD --ret_model_types HAE_R300_MAG_gpt2S HAE_R300_MAG_S_TD HAE_R300_MAG_H768_S HRE_G300_MAG_H768_S MN_G300_MAG_H768_S
```


## Notes
#### Teststd Results
Teststd results for task1 & task2 were in "simmc/mm_action_prediction/".

You can evaluate with provided codes.
#### Model parameters
We have all model parameters for generating devtest and teststd result. (trained with our entry code before Sep 28th)

If you need them, please contact us via email (md98765@naver.com)
