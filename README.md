# pfn_example

Simple repo that has pfn_example.py from energyflow, and a requirements file.

The energyflow documentation is, as always, found at : [https://energyflow.network/](https://energyflow.network/)

This example was originally written by Patrick T. Komiske III and Eric Metodiev. This is just a repackaged example with a few more instructions!

## Getting up and running

This setup is working at the Brown University Ocean State Centre for Advances Resources (OSCAR) high-performance computing cluster. It should work other places, too!

Follow these instructions:

If you are running on OSCAR, be sure to enter an interactive session before doing too much heavy lifting:

```
interact -q gpu -g 1 -m 32g -t 24:00:00
```

Then, from scratch,

```
# Make a work directory if you don't already have one set up
mkdir work
cd work/ 

# Set up a python virtual environment (only needed the first time)
python -m venv tensorflow.venv 

# Enter the environment (need to run in every shell you want to train)
source tensorflow.venv/bin/activate

# To install the dependencies (only needed the first time)
pip install --upgrade pip
pip install -r requirements.txt
```

If all goes well, then you should be able to run the example script:

```
python pfn_example.py
```

The first time you run it, it will download the q/g tagging dataset. If you just want to run a quick test, reduce the number of jets that are being processed (https://github.com/mattleblanc/pfn_example/blob/main/pfn_example.py#L55). I have modified this version of the example script to use early stopping instead of a fixed number of epochs.

