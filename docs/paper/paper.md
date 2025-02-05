---
title: 'Stresslog: A python package for modelling wellbore stability in inclined stress states'
tags:
    - Python
    - Drilling
    - Geomechanics
    - Pore pressure
    - Mud loss
    - Wellbore stability
authors:
    - name: Arkadeep Ghosh
    - orcid: 0009-0008-3209-8125
    corresponding: true
    affiliation: 1
affiliations:
    -name: Rock Lab Private Limited, India
    index: 1
date: 31 January 2025
bibliography: paper.bib
---

# Summary

This software is meant to be used by researchers and practitioners working in the field of geomechanics. It is a python library that uses a collection of algorithms used to iteratively model the sate of stress underground, given a well log in las format. The algorithms used are compatible with each other, and have been chosen to be applicable as universally as possible. The computations use full 6 component stress tensor (as calculated using [@pevska1995]) which allows modelling inclined wellbores as well as inclined states of stress. The program estimates, among other things, pore pressure (with and without unloading effects) from sonic, resistivity and d-exponent, principal stresses, hoop stresses, compressive strength, tensile strength, the three moduli of elasticity and sanding tendency. The D-exponent itself is calculated based on drilling data if available.

# Statement of need

It is possible to use nothing other than a standard spreadsheet to carry out geomechanical modelling, however such a process is very tedious if iterative methods are to be used.

Most software in standard use in the petroleum industry apply certain simplifying assumptions, mainly in the form of assuming that the vertical stress constitutes a principal stress. And that is usually a very good approximation, but there are situations where this is not the case, especially in regions experiencing isostatic re-adjustment or salt tectonism, among others. While there are open-source multi-physics packages, using them in a wellsite/operations sense, especially if real-time performance is required, is ill-advised. There is thus, a need for a dedicated tool for wellbore stability analysis.

This python package is aimed at empowering researchers with a simple to use and comprehensive 1D mechanical earth modelling tool that is freely available and which researchers can modify to apply their own methods when neccessary, while allowing practitioners to use the pre-defined algorithms to calculate solutions, iteratively process and export well log data.

# Methodology

Overburden gradient [@traugott1997], pore pressure [@Zhang20132] [@Flemings2021], minimum horizontal stress [@Daines1982] [@zoback1992], rock strength [@lal1999] [@horsrud2001] and other parameters are calculated considering the given well-logging, deviation, and formation data. The maximum horizontal stress is estimated by applying stress polygon [@ZOBACK2003] for every depth-sample. Borehole image interpretation is considered in the stress polygon results if available.

The calculation of tilted stress states using the given methodology requires the Euler angles Alpha, Beta and Gamma. However this is not immediately apparent from a geological perspective. We therefore calculate the Euler angles from geological data in terms of dip angle and dip azimuth.

The tilt of the stress tensor is calculated from dip angle and dip azimuth as follows:
Tilt Direction : ArcTan(Rs[2][1]/Rs[2][0])
Tilt Angle : ArcCos(Rs[2][2])
Where Rs is the rotation matrix defined by Euler angles Alpha, Beta and Gamma, in the NED reference frame. In particular,
Rs = [[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]]

To get the dip direction of the plane perpendicular to the tilt pole, we add 180 degrees to the tilt direction. In the program, the azimuth of maximum principal stress is specified, and the plane perpendicular to the tilt pole is specified using dip azimuth and dip angle. Then considering the azimuth of maximum principal stress as Alpha, the above relations are used to optimise for the angles Beta and Gamma. The Euler angles are then used for further stress transformations as required. Nelder Mead algorithm is used, taking care to avoid local minima.

In the technique proposed by [@pevska1995], they start with good estimates of the far field principal stresses, Sigma1, Sigma2 and Sigma3, already rotated by the Euler Angles Alpha, Beta and Gamma. During usual modelling however, what is available is an estimate of minimum horizontal stress, an estimate of the vertical stress (Sv, from overburden gradient), and perhaps an estimate of maximum horizontal stress. Given this situation, it is insufficient to simply rotate the tensor, as the rotated tensor will not have the correct vertical component. To remedy this, we optimise the principal stresses (sigma 1, sigma 2 and sigma 3) such that the vertical and horizontal components of the tensor match the specified horizontal and vertical stresses. L-BFGS-B is used as the algorithm for the minimisation. The optimized values are then used as the far field stress tensor after rotation by alpha, beta and gamma, for stability calculations.

For every depth-sample, the stresses resolved on the wellbore wall are calculated along the circumference at 10 degree intervals. The lower critical mudweight is calculated by using the modified Lade formula for critical mudweight during this process, the value for each sample is calculated from this array by taking a percentile value (thus if upto 90 degree width breakouts are tolerable then we take the 50 percentile value, the acceptable width of breakouts in degrees is set by the 'mabw' parameter). The upper critical mudweight (fracture gradient) can be calculated by itertatively minimising the difference between the sigma-theta-minimum (the minimum principal stress as resolved on the hole wall) and the tensile strength (here taken to be -UCS/10). Considering that the minimum principal stress along the wellbore wall is a function of pore pressure, minimum horizontal stress, maximum horizontal stress, overburden stress, thermal stress, euler angles alpha, beta and gamma, and wellbore inclination and azimuth, as well as the mud pressure at the given depth (providing the radial stress), we can express this as follows:
f(pore pressure, minimum horizontal stress, maximum horizontal stress, overburden stress, thermal stress, alpha, beta, gamma, inclination, azimuth, critical mud weight)-g(UCS) = 0
From this, knowing the exact expression of f and g, we solved for critical mud weight using sympy to arrive at a closed-form solution. This technique is faster than the minimization approach, and more robust in the sense that problems of local minima and numerical instability are avoided.

If the user specifies an analysis depth, then a orientation-stability plot is calculated for that depth, given the mud weight, stresses and pore pressure, as well as uniaxial compressive strength, poisson's ratio, temperature difference between borehole wall and circualting fluid, coefficient of thermal expansion and biot's poroelastic constant. Mohr-Coloumb failure criteria is used to predict compressive failures (borehole breakouts). For tensile failure, a simplified Griffith failure criteria is used. A synthetic image of the wellbore wall is prepared for 5 metres above and below the analysis depth. By comparing the output(s) with recorded well data, the user may change the model parameters to achieve better agreement between observed and calculated values. Other plots are also calculated for the analysis depth, including sanding prediction using [@willson2002] and [@Zhang2007].

For pre-drill forecast, the function get_analog() can be used to derive a log-prediction from nearby post-drill well given the welly.Well object representing the post-drill well, formation tops for the post-drill well and predicted formation tops as well as deviation data for the pre-drill well. The new logs are derived by depth-shift processing the post-drill logs as per the change in formation tops. The returned object can then be used for estimating the geomechanical properties in the regular manner.

# Case Study

The well data from Equinor Northern Lights dataset [northernlights] has been used as the example here, to model the stress state occuring in the Lower Drake formation, in the depth interval of 2600 to 2630m. The resistivity image log shows the occurance of en-echelon fractures in a vertical wellbore. The model applied here uses parameters very similar to [@Thompson2022], and a stress tensor tilt of 2 degrees towards south, and is able to replicate the fracture patterns observed in the actual image log.
![(a) Model of EOS Northern Lights Well showing the Drake I, II and IntraMarine formations. \label{fig:EOS_NorthernLights}](../Figures/WellPlot.png)
![(b) Fracture patterns calculated at 2624.5m and superimposed onto the image log. \label{fig:Fracture Motif}](../Figures/overlay.png)

It is not being suggested that this interpretation of the data is preferred over any other, this example is merely meant to show the capability of the package. In particular, the analysis by [@Thompson2022] likely offers a better explanation of simultaneous en-echelon breakouts and induced fractures as it is a far more detailed analysis than the example here.

# Discussion

From observations on multiple wells sampling the same stress field at different wellbore orientations, a better estimate of the stress tensor orientation is possible [@thorsen2011]. Further work in the future may help automate the process by incorporating the means to import the image log data directly, calculating a difference between the calculated image log and the imported one, and attempt to manipulate the stress orientation in an effort to minimise the difference. The current program is modular in nature, and the stability calculation subroutines can be re-used for this purpose.

Currently there are certain aspects which are not considered by the program, which would boost usefulness and accuracy if included. These include water-shale interactions and rock strength and elastic moduli anisotropy. We hope to implement these features in the future, but as these do not detract from the usability of the program as is, (and because data for calibrating these are somewhat rare), we have chosen to omit these at the current stage of development.

Not everyone is comfortable with python, for such users a webapp using stresslog, streamlit (and toga for offline use) is currently in development.

# Disclosure
No funding/financial support of any form was involved in the creation of this work.

# References