# BACS: Bundle Adjustment For Camera Systems

This is a Python implementation of BACS, a bundle adjustment for camera systems with points at infinity. It was originally written in Matlab and published by Johannes Schneider, Falko Schindler, Thomas Laebe, and Wolfgang Foerstner in 2012.

## Usage

Run

```bash
python3 -m pip install bacs
```

to install the library. 
Have a look at the [extensive doc string](https://github.com/zauberzeug/bacs/blob/main/bacs/bacs.py#L34-L77) for explenation of the parameters.

## Testing / Developing

Make sure you have NumPy and SciPy installed:

```bash
python3 -m pip install numpy scipy
```

By running the provided examples with

```bash
python3 main.py
```

you can verify that bacs is working correctly (eg. no `git diff` in the output data after execution).

## Resources

Further explanation and visualization can be found on the [BACS project page](https://www.ipb.uni-bonn.de/data-software/bacs/), the corresponding [Matlab demo](https://www.ipb.uni-bonn.de/html/software/bacs/v0.1/demo-v0.1.html) as well as the original [publication](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/I-3/75/2012/isprsannals-I-3-75-2012.pdf)
