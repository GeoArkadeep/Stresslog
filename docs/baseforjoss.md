---
title: 'Stresslog: A GUI for modelling wellbore stability in inclined stress states'
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

This software is meant to be used by researchers and practitioners working in the field of geomechanics. It is a python library that uses a collection of algorithms used to iteratively model the sate of stress underground, given a well log in las format. The algorithms used are compatible with each other, and have been chosen to be applicable as universally as possible. The computations use full 6 component stress tensor (as calculated using peska and zoback,[]) which allows modelling inclined wellbores as well as inclined states of stress. The program estimates, among other things, pore pressure (with and without unloading effects) from sonic, resistivity and d-exponent, principal stresses, hoop stresses, compressive strength, tensile strength, the three moduli of elasticity and sanding tendency. The D-exponent itself is calculated based on drilling data if available.

# Statement of need

Most research in the field of geomechanics either use proprietary software to carry out their research, or a combination of open source packages in piecemeal fashion. It is possible to use nothing other than a standard spreadsheet to carry out geomechanical modelling, however such a process is very tedious if iterative methods are to be used.

Most software in standard use in the petroleum industry apply certain simplifying assumptions, mainly in the form of assuming that the vertical stress constitutes a principal stress. And that is usually a very good approximation, but there are situations where this is not the case, especially near diapirs and regions experiencing lithospheric flexure, among others.

This software is aimed at empowering researchers with a simple to use and comprehensive 1D mechanical earth modelling tool that is freely available and which researchers can modify to apply their own methods when neccessary, while allowing practitioners to use the pre-defined algorithms to calculate solutions, iteratively process and export well log data.

# Methodology

The process requires a well log, along with associated deviation data, wellbore data (such as mudweights, casing shoe depths, bit diameter for different hole sections etc.), as well as other auxiliary data that the user may have, such as strenths from core analysis, pressure test and minifrac data, formation tops, observations from image logs, and the lithology column. The user has the option of specifying certain parameters such as poisson's ratio and UCS for each lithology, and many other geomechanical parameters for each formation, in the auxiliary data. The auxiliary data is not necessary for program execution, but if provided will help the user constrain the results better. We start by calculating overburden gradient[], pore pressure[], minimum horizontal stress[] and UCS[]. A permeability filter is applied to the pore pressure gradient data using the absolute difference between the deep and shallow resistivity logs, and permeable segments above a user specified cutoff are rejected, followed by interpolation of the pore pressure gradient data. The pressures are calculated by integrating the pore pressure gradient, minimum horizontal stress and overburden gradients. For permeable intervals, the hydraulic pore pressures are calculated considering the data available in the formation tops dataframe, such as formation tops (in TVD), structure top depths, oil-water contact depths, gas-oil contact depths, centroid ratio, oil gradients and water gradients. If any of the data are missing, meaningful defaults are used wherever possible (such as centroid ratio defaulting to 0.5). The maximum horizontal stress is estimated by applying stress polygon for every depth-sample, if borehole image interpretation is available, that is used to better constrain the upper and lower limits of maximum horizontal stress. 

The calculation of tilted stress states using the given methodology requires the Euler angles Alpha, Beta and Gamma. However this is not immediately apparent from a geological perspective. To solve this, we attempt to calculate the Euler angles from geological data in terms of dip angle and dip azimuth. While there are many formats of specifying geological data, such as strike direction and dip angle, we have chosen to specify the dip direction and dip angle, as they are the least ambiguous. It should be noted that the dip direction and angle being referred to here is that of the plane containing two of the principal stress axes, taken here to be the two which are the closest to the horizontal plane.

The tilt of the stress tensor is calculated from dip angle and dip azimuth as follows:
Tilt Direction : ArcTan(Rs[2][1]/Rs[2][0])
Tilt Angle : ArcCos(Rs[2][2])
Where Rs is the rotation matrix defined by Euler angles Alpha, Beta and Gamma, in the NED reference frame. In particular,
Rs = [[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]]

To get the dip direction of the plane perpendicular to the tilt pole, we add 180 degrees to the tilt direction. In the program, the azimuth of Sigma1 is specified, and the plane perpendicular to the tilt pole is specified using dip azimuth and dip angle. Then considering the azimuth of Sigma1 as Alpha, the above relations are used to optimise for the angles Beta and Gamma. The Euler angles are then used for further stress transformations as required. As no constraints are used for the minimisation, Nelson Mead algorithm is used for this calculation, taking care to avoid local minima.

In the technique proposed by Peska and Zoback, they start with good estimates of the far field principal stresses, Sigma1, Sigma2 and Sigma3, already rotated by the Euler Angles Alpha, Beta and Gamma. During usual modelling however, what is available is an estimate of minimum horizontal stress, an estimate of the vertical stress (Sv, from overburden gradient), and perhaps an estimate of maximum horizontal stress. Given this situation, it is insufficient to simply rotate the tensor, as the rotated tensor will not have the correct vertical component. To remedy this, we optimise the principal stresses (sigma 1, sigma 2 and sigma 3) such that the vertical and horizontal components of the tensor match the specified horizontal and vertical stresses. L-BFGS-B is used as the algorithm for the minimisation. The optimized values are then used as the far field stress tensor after rotation by alpha, beta and gamma, for stability calculations.

For every depth-sample, the stresses resolved on the wellbore wall are calculated along the circumference at 10 degree intervals. The lower critical mudweight is calculated by using the modified Lade formula for critical mudweight during this process, the value for each sample is calculated from this array by taking a percentile value (thus if upto 90 degree width breakouts are tolerable then we take the 50 percentile value, the acceptable width of breakouts in degrees is set by the 'mabw' parameter). The upper critical mudweight (fracture gradient) can be calculated by itertatively minimising the difference between the sigma-theta-minimum (the minimum principal stress as resolved on the hole wall) and the tensile strength (here taken to be -UCS/10). Considering that the minimum principal stress along the wellbore wall is a function of pore pressure, minimum horizontal stress, maximum horizontal stress, overburden stress, thermal stress, euler angles alpha, beta and gamma, and wellbore inclination and azimuth, as well as the mud pressure at the given depth (providing the radial stress), we can express this as follows:
f(pore pressure, minimum horizontal stress, maximum horizontal stress, overburden stress, thermal stress, alpha, beta, gamma, inclination, azimuth, critical mud weight)-g(UCS) = 0
From this, knowing the exact expression of f and g, we can solve for critical mud weight to arrive at a closed-form solution, which is what is used in the package. This technique is faster than the minimization approach, and more robust in the sense that problems of local minima and numerical instability are avoided.

If the user specifies an analysis depth, then a orientation-stability plot is displayed for that depth, given the mud weight, stresses and pore pressure, as well as uniaxial compressive strength and poisson's ratio (calculated or specified). Mohr-Coloumb failure criteria is used to predict compressive failures (borehole breakouts) using parameters derived from Lal's correlations[]. For tensile failure, a simplified Griffith failure criteria is used. A synthetic image of the wellbore wall is prepared for 5 metres above and below the analysis depth and this is displayed. By comparing the output(s) with recorded well data, the user may change the model parameters to achieve better agreement between observed and calculated values.

For pre-drill forecast, the function get_analog() can be used to derive a log-prediction from nearby post-drill well given the welly.Well object representing the post-drill well, formation tops for the post-drill well, predicted formation tops and deviation data for the pre-drill well. The returned object can then be used for estimating the geomechanical properties in the regular manner. The new logs are derived by depth-shift processing the post-drill logs as per the change in formation tops.

# Case Study

Two case studies are presented here for validation, one vertical well and one deviated. The well data from Equinor Northern Lights dataset has been used as the vertical well example here, to model the stress state occuring in the Lower Drake formation, in the depth interval of 2600 to 2630m. The resistivity image log shows the occurance of en-echelon fractures in a vertical wellbore. The model applied here uses parameters from Thompson et al[], and a stress tensor tilt of 2 degrees towards south, and is able to replicate the fracture patterns observed in the actual image log.
![(a) Model of EOS Northern Lights Well showing the Drake I, II and IntraMarine formations (b) Fracture patterns calculated at 2624.5m (c) Hoop stress and fracture angles at the same depth \label{fig:EOS_NorthernLights}](../media/EOS_NorthernLightsAandB.pdf)


# Discussion

It is accepted that this method of estimating stress tensor does not uniquely define the stress tensor orientation, as many different combinations of alpha, beta and gamma lead to similar stress states. Thus it is a known limitation that the program cannot be used to uniquely constrain the orientation of the stress state from a single well. However, from observations on multiple wells sampling the same stress field at different wellbore orientations, a better estimate of the stress tensor orientation is possible. The program does not currently operate on more than one well at a time, and the iterations to arrive at a satisfactory orientation (considering one or multiple wells) are carried out by visual estimation alone. Further work in the future may help automate the process by incorporating the means to import the image log data directly, calculating a difference between the calculated image log and the imported one, and attempt to manipulate the stress orientation in an effort to minimise the difference. The current program is modular in nature, and the stability calculation subroutines in the module BoreStab.py can be re-used for this purpose.

# Disclosure
No funding/financial support of any form was involved in the creation of this work. The author retains the rights to make available commercially, derivative forms of this software, in the future.

# References