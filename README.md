# CLOSEgaps

## Abstract
Incomplete knowledge of metabolic processes hinders the accuracy of GEnome-scale Metabolic models (GEMs), which in turn impedes advancements in systems biology and metabolic engineering. Existing gap-filling methods typically rely on phenotypic data to minimize the disparity between computational predictions and experimental results. However, there is still a lack of an automatic and precise gap-filling method for initial state GEMs before experimental data and annotated genomes become available. In this study, we introduce CLOSEgaps, a deep learning-driven tool that addresses the gap-filling issue by modeling it as a hyperedge prediction problem within GEMs. Specifically, CLOSEgaps maps metabolic networks as hypergraphs and learns their hyper-topology features to identify missing reactions and gaps by leveraging hypothetical reactions. This innovative approach allows for the characterization and curation of both known and hypothetical reactions within metabolic networks. Extensive results demonstrate that CLOSEgaps accurately gap-filling over $96\%$ of artificially introduced gaps for various GEMs. Furthermore, CLOSEgaps enhances phenotypic predictions for $24$ GEMs and also finds a notable improvement in producing four crucial metabolites (Lactate, Ethanol, Propionate, and Succinate) in two organisms. As a broadly applicable solution for any GEM, CLOSEgaps represents a promising model to automate the gap-filling process and uncover missing connections between reactions and observed metabolic phenotypes.

![image](./img/Fig1_new_v1.png)

## Dependencies
The package depends on the Python==3.7.13:
```
cobra==0.22.1
joblib==1.2.0
numpy==1.21.5
optlang==1.5.2
pandas==1.3.5
torch==1.12.1
torch_geometric==2.1.0
torch_scatter==2.0.9
torch_sparse==0.6.15 
tqdm==4.62.1
scikit-learn==1.0.2
rdkit==2022.03.5
```

## Datasets
We utilized CLOSEgaps to predict missing reactions in both metabolic networks and chemical reaction datasets. The detail of all datasets is shown as below:
| oprule Dataset | Species                                    | Metabolites (vertices) | Reactions (hyperlinks) |
|----------------|--------------------------------------------|------------------------|------------------------|
| Yeast8.5       | Saccharomyces cerevisiae (Jul. 2021)       | 1136                   | 2514                   |
| iMM904         | Saccharomyces cerevisiae S288C (Oct. 2019) | 533                    | 1026                   |
| iAF1260b       | Escherichia coli str.K-12 substr.MG1655    | 765                    | 1612                   |
| iJO1366        | Escherichia coli str.K-12 substr.MG1655    | 812                    | 1713                   |
| iAF692         | Methanosarcina barkeri str.Fusaro          | 422                    | 562                    |
| USPTO\_3k      | Chemical reaction                          | 6706                   | 3000                   |
| USPTO\_8k      | Chemical reaction                          | 15405                  | 8000                   |

The datasets are stored in ```./data``` and each contains reactions and metabolites' SMILES. 
For example, 
* The folder ```./data/yeast```  contains yaset dataset. 
* The file ```./data/yeast/yeast_rxn_name_list.txt``` contains the reactions.
* The file ```./data/yeast/yeast_meta_count.csv``` contains each metabolic's name, SMILES, and atom number.

## Processing GEMs
Mapping GEMs to SMILES with public database:
```bash
$ python GEM_process.py --GEM_name yeast
```

## Running the Experiment
To run our model in the yeast dataset, based on the default conditions, which set the ratio of positive and negative reactions as 1:1, imbalanced atom number, and the ratio of replaced atoms for negative reaction as 0.5:
```bash
$ python main.py
```
If you want to run our model based on different creating negative samples strategies, run the following script:
```bash
$ python main.py --train yeast --output ./output/ --create_negative True --balanced True --atom_ratio 0.5 --negative_ratio 2
```

<kbd>train</kbd> specifies the training dataset (For example, ```yeast```, ```uspto_3k```,  ```iMM904```, and so on).

<kbd>output</kbd> specifies the path to store the model.

<kbd>create_negative</kbd> specifies whether to create negative samples based on different conditions. If <kbd>False</kbd>, the model will run on the default train, valid, and test data, and when <kbd>True</kbd>, you need to set other parameters to create negative samples. 

<kbd>balanced</kbd> specifies whether to replace metabolic based on balanced atom number.

<kbd>atom_ratio</kbd> specifies the ratio of replaced atoms for negative reaction.

<kbd>negative_ratio</kbd> specifies the ratio of negative reaction samples.

Use the command <code> python main.py -h </code>to check the meaning of other parameters.

# GEMs reconstruction workflow
All input files should be stored in the data directory. This directory contains three sub-folders:

## data/gems
This folder contains the GEMs that will be tested. Each GEM is saved as an XML file.

## data/pools
This folder contains the hypothetical reaction, named `universe.xml`. To use your own pool, rename it to `universe.xml and update the `EX_SUFFIX` and `NAMESPACE` parameters in the `input_parameters.txt` file.

## data/fermentation
The file `substrate_exchange_reactions.csv` contains a list of fermentation compounds that will be searched for missing phenotypes in the input GEMs. Additionally, the file `media.csv` specifies the culture medium used to simulate the GEMs.
  
## Simulation Parameters
All simulation parameters are defined in the `input_parameters.txt`.

1. Score the candidate reactions in the pool for their likelihood of being missing in the input GEMs (function `predict()` in `main.py`).

2. Among the top candidate reactions with the highest likelihood, find out the minimum set that leads to new metabolic secretions that are potentially missing in the input GEMs (function validate() in `fba` folder's `main.py`). The second program is time-consuming if the number of top candidates added to the input GEMs for simulations is too large (this parameter is controlled by `NUM_GAPFILLED_RXNS_TO_ADD` in the `input_parameters.txt`).

## Pretrained Model
You can access the pretrained model for CLOSEgaps at the following link, CLOSEgaps Pretrained Model: https://zenodo.org/records/13691968. Feel free to download and use it for testing and further experimentation.
