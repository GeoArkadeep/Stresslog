# Well Master GeoMech

A *libre* application to calculate and plot pore pressure and other geomechanical data from las files. It is intended for pre-drill and post-drill studies of wells, and can be used for rudimentary semi-realtime stress prediction (for example if the wellsite geologist fetches and updates the las from MWD/LWD provider every single or so). A Pro version with improved features, including true realtime capabilities from streaming data, will be made available seperately under a commercial licence later on.

## Features:

* Import las files, deviation files, ucs and lithology files

* Alias logs using user modifiable alias file

* Pad log data all the way to surface, using appropriate KB, GL or WD values (Option to specify these and correct these in the Log Header)

* Auto export the well data as .csv, so you can run your own plots and calculations in your software of choice

* Calculate Pore Pressure, Shmin, SHMax, Sv and UCS from DT

* Model saved to and read from model.csv, calibrate model on analog well and run it on current well using sonic velocity from seismic and/or LWD/MWD data 

* Stress Polygon, Wellbore Stability Plots and Kirsch Plots at drainhole analysis depth

* Plot the data in user configurable intervals

* Option to save Plots ~~at custom DPI~~

* Option to output Las Files (with updated header and new data columns)

* Fields for adding mud data, loss/gain data and other data interpreted from drilling and testing history, these are plotted over the calculated data to help visually constrain the model

* Option to include ascii file with observations from resistivity image logs (or others like ultrasonic calipers) to include tensile fractures and breakouts, which are then used to automatically constrain SHMax

## Installation:

You can download the setup file from the release section, and install it like any normal program.

Or, you can compile from source. To compile from source, you need toga and briefcase packages in your python environment.
````
pip install toga
pip install briefcase
````
Once these are installed, you can build using the briefcase commands
````
briefcase create
briefcase build
briefcase package
````

For more help, consult the BeeWare documentation.

## Acknowledgements

This software is written in Python, using the python library Welly by Agile Scientific, Canada. Welly itself uses the Lasio library to handle the las files. The GUI is written using Toga, and built using Briefcase, which are both components of the BeeWare Project. Other libraries used include Pandas, Numpy and Matplotlib.

## Disclaimer

IN MAKING INTERPRETATIONS OF LOGS THIS SOFTWARE AND ITS AUTHOR(S) WILL GIVE USERS THE BENEFIT OF THEIR BEST JUDGEMENT. BUT SINCE ALL INTERPRETATIONS ARE OPINIONS BASED ON INFERENCES FROM ELECTRICAL OR OTHER MEASUREMENTS, WE CANNOT, AND WE DO NOT GUARANTEE THE ACCURACY OR CORRECTNESS OF ANY INTERPRETATION. WE SHALL NOT BE LIABLE OR RESPONSIBLE FOR ANY LOSS, COST, DAMAGES, OR EXPENSES WHATSOEVER INCURRED OR SUSTAINED BY THE USER RESULTING FROM ANY INTERPRETATION MADE BY THE SOFTWARE OR ITS AUTHOR(S).

THERE IS NO WARRANTY FOR THE PROGRAM. THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

THE STATEMENTS ABOVE IS IN ADDITION TO THE CONCERNED SECTIONS OF THE AGPL3.0 LICENSE GOVERNING THE PROGRAM.
