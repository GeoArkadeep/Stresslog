---
title: "Stresslog: A python package for modelling wellbore stability in inclined stress states"
tags:
  - Python
  - Drilling
  - Geomechanics
  - Pore pressure
  - Mud loss
  - Wellbore stability
authors:
  - name: Arkadeep Ghosh
    orcid: 0009-0008-3209-8125
    corresponding: true
    affiliation: 1

affiliations:
  - name: Rock Lab Private Limited, India
    index: 1

date: 10 February 2025
bibliography: paper.bib
---

# Summary

This package is meant to be used by researchers and practitioners working in the field of geomechanics. It uses a collection of algorithms used to iteratively model the sate of stress underground, given a well log. The computations use full 6 component stress tensor (as calculated using [@pevska1995]) which allows modelling inclined wellbores as well as inclined states of stress. Stresslog estimates, among other things, pore pressure (with and without unloading effects) from sonic, resistivity and d-exponent, principal stresses, hoop stresses, compressive strength, tensile strength, the three moduli of elasticity and sanding tendency.

# Statement of need

Stresslog has been designed to help with pre-drill, post-drill and realtime geomechanical calculations.
It is often assumed that the vertical stress constitutes a principal stress which is usually a very good approximation. There are situations where this is not the case, especially in regions experiencing isostatic re-adjustment or salt tectonism, among others. Stresslog is aimed at empowering researchers with a simple to use and comprehensive 1D mechanical earth modelling tool that is freely available and which researchers can modify to apply their own methods when neccessary, while allowing practitioners to use the pre-defined algorithms to calculate solutions, iteratively process and export well log data.

# Methodology

Overburden gradient [@traugott1997], pore pressure [@Zhang20132] [@Flemings2021], minimum horizontal stress [@Daines1982] [@zoback1992], rock strength [@lal1999] [@horsrud2001] and other parameters are calculated considering the given well-logging, deviation, and formation data. The maximum horizontal stress is estimated by applying stress polygon [@ZOBACK2003] for every depth-sample. Borehole image interpretation is considered in the stress polygon results if available.

The calculation of tilted stress states using the given methodology requires the Euler angles Alpha, Beta and Gamma. However this is not immediately apparent from a geological perspective. We therefore calculate the Euler angles from geological data in terms of tilt azimuth and tilt angle of the stress tensor.

The tilt of the stress tensor is calculated from dip angle and dip azimuth as follows:

$$
\text{Tilt Azimuth} = \tan^{-1}\left(\frac{R_{s_{3,2}}}{R_{s_{3,1}}}\right)
$$

$$
\text{Tilt Angle} = \cos^{-1}(R_{s_{3,3}})
$$

Where $R_s$ is the rotation matrix defined by Euler angles $\alpha$, $\beta$ and $\gamma$, in the NED reference frame:

$$
R_s = \begin{bmatrix} 
\cos\alpha\cos\beta & \sin\alpha\cos\beta & -\sin\beta \\
\cos\alpha\sin\beta\sin\gamma - \sin\alpha\cos\gamma & \sin\alpha\sin\beta\sin\gamma + \cos\alpha\cos\gamma & \cos\beta\sin\gamma \\
\cos\alpha\sin\beta\cos\gamma + \sin\alpha\sin\gamma & \sin\alpha\sin\beta\cos\gamma - \cos\alpha\sin\gamma & \cos\beta\cos\gamma
\end{bmatrix}
$$

Considering the azimuth of maximum principal stress as $\alpha$, the above relations are used to optimise for the angles $\beta$ and $\gamma$. The Euler angles are then used for further stress transformations.

In the technique proposed by [@pevska1995], they start with good estimates of the far field principal stresses, $\sigma_1$, $\sigma_2$ and $\sigma_3$, already rotated by the Euler Angles $\alpha$, $\beta$ and $\gamma$. Usually, however, what is available is an estimate of minimum horizontal stress, an estimate of the vertical stress, and an estimate of maximum horizontal stress. Given this, it is insufficient to simply rotate the tensor, as the rotated tensor will not have the correct vertical component. To remedy this, we optimise the principal stresses ($\sigma_1$, $\sigma_2$ and $\sigma_3$) such that the vertical and horizontal components of the tensor match the specified horizontal and vertical stresses.

For every depth-sample, the stresses resolved on the wellbore wall are calculated along the circumference at 10 degree intervals. The lower critical mudweight is calculated by using the modified Lade formula for critical mudweight during this process, the value for each sample is calculated from this array by taking a percentile value. A closed-form solution has been derived by setting $\sigma_{\theta_{\min}}$ equal to tensile stress and solving this for the upper critical mud pressure, as follows:

$\text{FracturePressure}_{\text{non-penetrating}} =$
$$
\frac{
  \left(
  \begin{aligned}
    &\quad +2\, \sigma_{B_{1,1}}'^2\, \nu\, \cos(2\theta_{min})
      - 4\, \sigma_{B_{1,1}}'^2\, \nu\, \cos(2\theta_{min})^2
      + 4\, \sigma_{B_{1,1}}'\, \sigma_{B_{1,2}}'\, \nu\, \sin(2\theta_{min})\\[1mm]
    &\quad - 8\, \sigma_{B_{1,1}}'\, \sigma_{B_{1,2}}'\, \nu\, \sin(4\theta_{min})
      + 8\, \sigma_{B_{1,1}}'\, \sigma_{B_{2,2}}'\, \nu\, \cos(2\theta_{min})^2
      + 2\, \sigma_{B_{1,1}}'\, \sigma_{B_{3,3}}'\, \cos(2\theta_{min})\\[1mm]
    &\quad - \sigma_{B_{1,1}}'\, \sigma_{B_{3,3}}'
      + 2\, \sigma_{B_{1,1}}'\, \nu\, PP\, \cos(2\theta_{min})
      - 2\, \sigma_{B_{1,1}}'\, \nu\, \sigma_T\, \cos(2\theta_{min})\\[1mm]
    &\quad - 2\, \sigma_{B_{1,1}}'\, \nu\, TS\, \cos(2\theta_{min})
      - 2\, \sigma_{B_{1,1}}'\, TS\, \cos(2\theta_{min})
      + \sigma_{B_{1,1}}'\, TS\\[1mm]
    &\quad - 16\, \sigma_{B_{1,2}}'^2\, \nu\, \sin(2\theta_{min})^2
      + 4\, \sigma_{B_{1,2}}'\, \sigma_{B_{2,2}}'\, \nu\, \sin(2\theta_{min})
      + 8\, \sigma_{B_{1,2}}'\, \sigma_{B_{2,2}}'\, \nu\, \sin(4\theta_{min})\\[1mm]
    &\quad + 4\, \sigma_{B_{1,2}}'\, \sigma_{B_{3,3}}'\, \sin(2\theta_{min})
      + 4\, \sigma_{B_{1,2}}'\, \nu\, PP\, \sin(2\theta_{min})
      - 4\, \sigma_{B_{1,2}}'\, \nu\, \sigma_T\, \sin(2\theta_{min})\\[1mm]
    &\quad - 4\, \sigma_{B_{1,2}}'\, \nu\, TS\, \sin(2\theta_{min})
      - 4\, \sigma_{B_{1,2}}'\, TS\, \sin(2\theta_{min})
      + 4\, \sigma_{B_{1,3}}'^2\, \sin(\theta_{min})^2\\[1mm]
    &\quad - 4\, \sigma_{B_{1,3}}'\, \sigma_{B_{2,3}}'\, \sin(2\theta_{min})
      - 4\, \sigma_{B_{2,2}}'^2\, \nu\, \cos(2\theta_{min})^2
      - 2\, \sigma_{B_{2,2}}'^2\, \nu\, \cos(2\theta_{min})\\[1mm]
    &\quad - 2\, \sigma_{B_{2,2}}'\, \sigma_{B_{3,3}}'\, \cos(2\theta_{min})
      - \sigma_{B_{2,2}}'\, \sigma_{B_{3,3}}'
      - 2\, \sigma_{B_{2,2}}'\, \nu\, PP\, \cos(2\theta_{min})\\[1mm]
    &\quad + 2\, \sigma_{B_{2,2}}'\, \nu\, \sigma_T\, \cos(2\theta_{min})
      + 2\, \sigma_{B_{2,2}}'\, \nu\, TS\, \cos(2\theta_{min})
      + 2\, \sigma_{B_{2,2}}'\, TS\, \cos(2\theta_{min})\\[1mm]
    &\quad + \sigma_{B_{2,2}}'\, TS
      + 4\, \sigma_{B_{2,3}}'^2\, \cos(\theta_{min})^2
      - \sigma_{B_{3,3}}'\, PP\\[1mm]
    &\quad + \sigma_{B_{3,3}}'\, \sigma_T
      + \sigma_{B_{3,3}}'\, TS
      + PP\, TS
      - \sigma_T\, TS
      - TS^2
  \end{aligned}
  \right)
}{
  2\, \sigma_{B_{1,1}}'\, \nu\, \cos(2\theta_{min})
  + 4\, \sigma_{B_{1,2}}'\, \nu\, \sin(2\theta_{min})
  - 2\, \sigma_{B_{2,2}}'\, \nu\, \cos(2\theta_{min})
  - \sigma_{B_{3,3}}'
  + TS
}
$$

where $\sigma'_B$ is the effective stress tensor in the borehole frame of reference, PP is pore pressure, TS is tensile strength, $\nu$ is Poisson's ratio and $\theta_{\min}$ is the circumferential angle corresponding to minimum hoop stress.

If the user specifies an analysis depth, then a orientation-stability plot is calculated for that depth. Mohr-Coloumb failure criteria is used to predict compressive failures. For tensile failure, Griffith failure criteria is used. A synthetic image of the wellbore wall is prepared for 5 metres around the analysis depth. Other plots are also calculated for the analysis depth, including sanding prediction using [@willson2002] and [@Zhang2007].

For pre-drill forecast, the function get_analog() can be used to derive a log-prediction from nearby post-drill well by interpolation using formation tops.

From observations on multiple wells sampling the same stress field at different wellbore orientations, a better estimate of the stress tensor orientation is possible [@thorsen2011].

# Case Study

The well data from Equinor Northern Lights dataset [northernlights] has been used as the example here, to model the stress state occuring in the depth interval of 2600 to 2630m. The resistivity image log shows the occurance of en-echelon fractures in a vertical wellbore. The model applied here uses parameters very similar to [@Thompson2022], and a stress tensor tilt of 2 degrees towards south, and is able to replicate the fracture patterns observed in the actual image log.

![Model of Northern Lights Eos Well showing the Drake I, II and IntraMarine formations. The sharp change at 2638m is due to thermal stresses not being considered below this depth.](../Figures/WellPlot.png)

![Fracture motifs calculated at 2624.5m and superimposed onto the image log. A perfectly vertical wellbore has been assumed in this example.](../Figures/overlay.png)

It is not being suggested that this interpretation of the data is preferred over any other, the analysis by [@Thompson2022] is much more comprehensive.

# Disclosure
No funding/financial support of any form was involved in the creation of this work.

# References