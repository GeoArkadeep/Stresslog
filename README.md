
![BG1](https://github.com/GeoArkadeep/WellMasterGeoMech/assets/160126374/27efa304-817d-4520-a5f5-65e63e679c15)
# Stresslog

A package to calculate and plot pore pressure and other geomechanical data from las files. It is intended for pre-drill and post-drill studies of wells, and can be used for rudimentary realtime stress prediction.

## Features

* Import las files, deviation files, ucs and lithology files

* Alias logs using user modifiable alias file

* Pad log data all the way to surface, using appropriate KB, GL or WD values (Option to specify these and correct these in the Log Header)

* Auto export the well data as .csv, so you can run your own plots and calculations in your software of choice

* Calculate Pore Pressure, Shmin, SHMax, Sv and UCS from DT

* Model saved to and read from model.csv, calibrate model on analog well and run it on current well using sonic velocity from seismic and/or LWD/MWD data 

* Plot Stress Polygon, Wellbore Stability ~~and Kirsch Plots~~ at drainhole analysis depth

* Plot the data in user configurable intervals

* Option to save Plots ~~at custom DPI~~ at 1200 dpi

* Option to output Las Files (with updated header and new data columns)

* Fields for adding mud data, loss/gain data and other data interpreted from drilling and testing history, these are plotted over the calculated data to help visually constrain the model

* Option to include ascii file with observations from resistivity image logs (or others like ultrasonic calipers) to include tensile fractures and breakouts, which are then used internally to better constrain SHMax

* Override nu, mu, and UCS from the lithology file 

## Installation


````
pip install stresslog
````
Consult the documentation for more help.


## Acknowledgements

Jon Jincai Zhang, for his awesome work on pore pressure prediction, and geomechanics at large. This software uses his algorithms wherever possible.

In more detail:
The Stability Analysis Plots are after [Peska and Zoback, 1995](https://doi.org/10.1029/95JB00319). The Stress Polygon Technique for constraining SHMax is after [Zoback et al., 2003](https://doi.org/10.1029/95JB00319) The Pore Pressure equation is from [Zhang, 2011](https://doi.org/10.1016/j.earscirev.2011.06.001) The shmin is calculated in accordance with [Dianes, 1982](https://doi.org/10.2118/9254-PA), as well as [Zoback and Healy,1992](https://doi.org/10.1029/91JB02175)

This software is written in Python, using the python library Welly by Agile Scientific, Canada. Welly itself uses the Lasio library to handle the las files. The GUI is written using Toga, and built using Briefcase, which are both components of the BeeWare Project. Other libraries used include Pandas, Numpy, Matplotlib and PIL.

## Disclaimer

IN MAKING INTERPRETATIONS OF LOGS THIS SOFTWARE AND ITS AUTHOR(S) WILL GIVE USERS THE BENEFIT OF THEIR BEST JUDGEMENT. BUT SINCE ALL INTERPRETATIONS ARE OPINIONS BASED ON INFERENCES FROM ELECTRICAL OR OTHER MEASUREMENTS, WE CANNOT, AND WE DO NOT GUARANTEE THE ACCURACY OR CORRECTNESS OF ANY INTERPRETATION. WE SHALL NOT BE LIABLE OR RESPONSIBLE FOR ANY LOSS, COST, DAMAGES, OR EXPENSES WHATSOEVER INCURRED OR SUSTAINED BY THE USER RESULTING FROM ANY INTERPRETATION MADE BY THE SOFTWARE OR ITS AUTHOR(S).

THERE IS NO WARRANTY FOR THE PROGRAM. THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

THE STATEMENTS ABOVE ARE IN ADDITION TO THE CONCERNED SECTIONS OF THE AGPL3.0 LICENSE GOVERNING THE PROGRAM.
