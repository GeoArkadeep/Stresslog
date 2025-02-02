Stresslog Documentation
=======================

Introduction
============

This is a python package for geomechanics. The core priorities of this package are:

- Being as accurate as possible:
    - Calculations take into account thermal stresses and Biot's poroelastic constant, and many of the parameters can be changed on per formation basis or per lithology basis
- Being as comprehensive as possible:
    - All sorts of parameters are calculated, using full 6 component stress tensor, calculated at all depth samples for a well, at 10 degree intervals on the bore wall circumference
- Being as automatic as possible:
    - All parameters other than the welly well object are optional, and include default values. We actively try to make the defaults such that the user need to alter as little as possible. While there are perhaps hundreds of algorithms, with more coming out everyday, we include only the author's curated favourites.

The functions shown here can be called independently, but the preferred way to use this package is using the compute_geomech function which calculates geomechanical parameters of an entire well at once.

We hope that this becomes *the* python package for geomechanics.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Introduction
   Examples
   API Reference
