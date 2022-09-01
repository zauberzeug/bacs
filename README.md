# BACS: Bundle Adjustment For Camera Systems

This is a Python implementation of BACS, a bundle adjustment for camera systems with points at infinity.
It was originally written in Matlab and published by Johannes Schneider, Falko Schindler, Thomas Läbe, and Wolfgang Förstner in 2012.

## Usage

Run

```bash
python3 -m pip install bacs
```

to install the library.
Have a look at the [doc string](https://github.com/zauberzeug/bacs/blob/main/bacs/bacs.py#L47-L92) for explanation of the parameters.

## Testing and development

Make sure you have NumPy and SciPy installed:

```bash
python3 -m pip install numpy scipy
```

By running the provided examples with

```bash
python3 main.py
```

you can verify that BACS is working correctly (eg. there is no `git diff` in the results folder after execution).

## Resources

Further explanation and visualization can be found on the [BACS project page](https://www.ipb.uni-bonn.de/data-software/bacs/), the corresponding [Matlab demo](https://www.ipb.uni-bonn.de/html/software/bacs/v0.1/demo-v0.1.html) as well as the original [publication](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/I-3/75/2012/isprsannals-I-3-75-2012.pdf).
