[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8730180&assignment_repo_type=AssignmentRepo)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py


## Module 2.5:
### Dataset: Simple  
- __data points__: 50
- __hidden layers__: 2
- __learning rate__: 0.1
- __epochs__: 500

The classification of the simple dataset is linear, hence a simple architecture with just two layers is enough. I kept the learning rate small to allow for a smooth convergence with no loss oscillation.

<p float="left">
        <img src="./imgs/simple_graph.png" height="500"/>
        <img src="./imgs/simple_terminal.png" height="500"/>
</p>
<img src="./imgs/simple_loss.png" height="800"/>

### Dataset: Diag  
- __data points__: 50
- __hidden layers__: 4
- __learning rate__: 0.5
- __epochs__: 250

The diag classification task is still linear but a bit less trivial than the one above. Therefore I increased the number of hidden layers to 4. A learning rate of 0.5 still allows for a smooth convergence with no big oscillation in loss and decrease the epochs number.

<p float="left">
        <img src="./imgs/diag_graph.png" height="500"/>
        <img src="./imgs/diag_terminal.png" height="500"/>
</p>
<img src="./imgs/diag_loss.png" height="800"/>

### Dataset: Split  
- __data points__: 50
- __hidden layers__: 6
- __learning rate__: 0.5
- __epochs__: 400

The split dataset is a bit more difficult to learn. I decided for a slightly deeper architecture with 6 layers and keeping a small learning rate of 0.5 to have smooth convergence with no loss oscillation, but that required a bit more epochs to train compared to diag.

<p float="left">
        <img src="./imgs/split_graph.png" height="500"/>
        <img src="./imgs/split_terminal.png" height="500"/>
</p>
<img src="./imgs/split_loss.png" height="800"/>


### Dataset: Xor  
- __data points__: 50
- __hidden layers__: 9
- __learning rate__: 0.5
- __epochs__: 550

This is the most difficult dataset to learn. I choose 9 layers but kept the learning rate to 0.5 since it was big enough to eventually converge while keeping the oscillation small.

<p float="left">
        <img src="./imgs/xor_graph.png" height="500"/>
        <img src="./imgs/xor_terminal.png" height="500"/>
</p>
<img src="./imgs/xor_loss.png" height="800"/>
