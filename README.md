# Hi-Viscont
The public repository of the paper Interactive Visual Task Learning for Robots

Interactive Visual Task Learning for Robots

Weiwei Gu, Anant Sah, and Nakul Gopalan

AAAI 2024

[Paper](https://arxiv.org/pdf/2312.13219.pdf) [Website](https://sites.google.com/view/ivtl) 

## Starting on the code base
### Prerequisites
+ Linux
+ Python3
+ PyTorch 1.6 with CUDA support
+ Other required python packages specified by `requirements.txt`.

### Installation
1. Clone this repository

    ```bash
    git clone https://github.com/WeiweiGu1998/HiViscont
    cd HiViscont
    ```
1. Create a conda environment for HiViscont and install the requirements. 
    
    ```bash
    conda create --n HiViscont
    conda activate HiViscont
    conda install pytorch=1.6.0 cuda100 -c pytorch #Assume you use cuda version 10.0
    pip install -r requirements.txt
    ```
1. Change `DATASET_ROOT` in `tools.dataset_catalog` to the folder where the datasets are stored. 
    Download and unpack the base CUB and our customized dataset datasets into 
    `DATASET/CUB-200-2011`,  `DATASET/test_dataset`,  respectively. 
    You can download images and corresponding annotations for CUB dataset from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).
    You can download model weights, everything for our customized dataset, and the questions generated for CUB dataset from [our webpage](https://sites.google.com/view/ivtl)
    
### Evaluation
1. Evaluate downloaded checkpoints with the config file(using a CUB configuration as example)
```bash
    export PYTHONPATH=${PYTHONPATH}:.
    python tools/test_concept_net.py --config-file experiments/cub/cub_fewshot_hierarchy_box_100.yaml
```

Notice that the numbers from the paper are the statistics from multiple evaluations, none of the checkpoint gives the exact number but should fall in the distribution.

### Training
1. We use CUB dataset as an example, assuming that you have downloaded the corresponding datasets and put everything into the correct folders.

2. First, pre-train the visual feature extractors by running:
```bash
    export PYTHONPATH=${PYTHONPATH}:.
    python tools/pretrain_net.py --config-file experiments/cub/cub_warmup_box_100.yaml
```
3. Train the few-shot concept learner by running:
```bash
    export PYTHONPATH=${PYTHONPATH}:.
    python tools/train_hierarchy_net.py --config-file experiments/cub/cub_fewshot_hierarchy_box_100.yaml
```

Scripts for training models for the robotics pipeline are also left in the codebase. 
Please feel free to contact the authors for more details or explanations if anything remains unclear.
