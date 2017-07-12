# convgp
Code for running Gaussian processes with convolutional and symmetric structures.

## Setup
### GPflow with inter-domain support
In order to run the examples here, you need to have a branch of GPflow installed that supports inter-domain inducing
variables. The branch located [here](https://github.com/markvdw/GPflow-inter-domain) and can be installed following the
usual GPflow instructions. You can either use your favourite virtualenv to install the branch in, or switch back and
forth between the main and customised versions of GPflow by running their setup script. 

### Datasets
You will also need to setup the datasets. In `/datasets`, run:
 ```
 python process_cifar10.py
 python process_rectangles.py
 ```

## Experiments
### Speed considerations
Running on the GPU is possible, and often significantly faster when using `float32` data. This reduced precision is fine
when using stochastic optimisation, but often problematic when using (variants of) BFGS. We recommend `float32` to be
used only with stochastic optimisation, and `float64` otherwise. This has to be adjusted manually in the `gpflowrc`file.

### Running experiments
We have the following experiments:
 - `rectangles.py`: rectangles dataset.
 - `vanilla_mnist.py`:
 
Many experiments have several command line options that can be used to modify a run. All have `--name`, which determines
the name of the file in `./results/` that stores the optimisation trace. Experiments are resumed if a file of the
correct name exists. Other options change the learning rate or minibatch size.

Optimisation traces can be displayed using `display.py`.

## Reproducing the plots in the paper
And some other interesting comparisons.

### Rectangles
```
python rectangles.py -k conv -M 16 --minibatch-size 100 -l 0.01 -n rectangles-paper  # Paper
python rectangles.py -k rbf -M 1200  # Compare to the optimal RBF
python rectangles.py -k wconv -M 16 --minibatch-size 100 -l 0.01  # Full solution
python rectangles.py -k conv -M 35 --minibatch-size 100 -l 0.01  # Full support on the GP
python rectangles.py -k wconv -M 35 --minibatch-size 100 -l 0.01  # Idem
```

### Mnist 0 vs 1
```
python rectangles.py -k wconv -M 16 --minibatch-size 100 -l 0.01
```

## Paper
