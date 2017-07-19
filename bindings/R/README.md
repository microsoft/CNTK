# R Interface to Microsoft Cognitive Toolkit

An R package for CNTK using Reticulate to bind to Python interface. Since it
binds directly to Python, the R interface can perform any operation the Python
bindings can, including running on the GPU. See the
[example notebook](cifar10-example.ipynb) to see a basic example of training
and eval of image classification the CIFAR-10 dataset.

#### Installation

To use CNTK with R you'll need to have Python with CNTK for Python already
installed on your machine. See
[this guide](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine)
for help setting up CNTK with Python.

Then run the following to install CNTK's R package:

    devtools::install_github("joeddav/CNTK/bindings/R") # will change to Microsoft/CNTK...

#### Usage

Documentation is still in the process, but the R package closely follows the
CNTK Python interface where possible
([Python docs](https://www.cntk.ai/pythondocs/index.html)). Here's the basic
rundown of the differences:

1. Properties are the same as in Python, and are accessed using the dollar sign
   ($) syntax:

```R
l <- Learner(parameters, lrschedule)
l$parameters # returns parameters associated with learner
```

2. Class methods are made global, and take the class object as the first
   property:

```R
learner.update(...) # Python
update_learner(learner, ...) # R equivalent
learner %>% update_learner(...) # R equivalent via pipe
```
As you can see, since class methods are made global, some renaming from the
original python was necessary to avoid conflicts. See [NAMESPACE](NAMESPACE) or
the [source code](R/) for a list of all functions while documentation is being
written.

3. R matrices are automatically converted to and from NumPy array's with
   float32 dtype.
4. Python enums are accessed via function argument, e.g.:

```R
UnitType.Error # Python
UnitType("Error") # R equivalent
```
