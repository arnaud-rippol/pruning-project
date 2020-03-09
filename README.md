# MAP583 Pruning Course Project
### Authors: Arnaud Rippol, Tarek Ayed, Florian Abeillon, Alban Jourdain, Zhuoan Ma

This project consists of implementing, testing and visualizing pruning techniques applied to standard deep learning models using the PyTorch library.

```test_norm_pruning.py``` allows to test and visualize pruning results for a given pruning method and a given model.

Here is a sample resulting plot with LeNet and a global unstructured L1 pruning approach:
![alt text](https://github.com/arnaud-rippol/pruning-project/blob/master/figures/pruning_results_LeNet_global_unstructured.png "Logo Title Text 1")

Arguments: 
```test_norm_pruning.py --model --model_path --dataset --verbose --show_plot --n_epochs --download --data_dir --batch_size --test_batch_size --crop_size --custom_transforms --use_cuda --save_model --save_to --lr --optimizer --loss --pruning_rates --pruning_methods```
Use ```test_norm_pruning.py --help``` to get more details about each argument.

```test_sparsity.py``` allows to test and visualize multiplication time of sparse and dense matrixes depending on sparsity rate. The point is to get a quantitative understanding of the potential impact of pruning on inference and backpropagation runtime.

Here is a sample resulting plot with 10 iterations and default sparsity array:
![alt text](https://github.com/arnaud-rippol/pruning-project/blob/master/figures/test_sparsity0.png "Logo Title Text 1")

Arguments: 
```test_sparsity.py --size [size of matrixes to generate] --iterations [number of iterations for each sparsity value] --sparsities [array of sparsities to test]```
All arguments are optional

```trainer.py``` is a script that computes accuracy values after training for a given model, either by loading a pre-trained version if it exists, or by training it from scratch.
