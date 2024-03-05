#Well Master GeoMech
===================

An *libre* application to calculate and plot pore pressure using p-wave velocity from las files. It is intended for pre-drill and post drill studies of wells, and can be used for rudimentary semi-realtime pressure prediction (for example if the wellsite geologist fetches and updates the las from MWD/LWD provider every single or so). A pro version with improved features, including true realtime capabilities from streaming data, will be made available seperately under a commercial licence later on.

##Features:

Import las files, deviation files, ucs and lithology files

Alias logs using user modifiable alias file

Pad log data all the way to surface, using appropriate KB, GL or WD values (Option to specify these and correct these in the Log Header)

Calculate Pore Pressure, shmin, SHMax, Sv and UCS from DT

Plot the data in user configurable intervals

Stress Polygon, Wellbore Stability Plots and Kirsch Plots at drainhole analysis depth

Option to save Plots ~~at custom DPI~~

Option to output Las Files (with updated header and new data columns)

Fields for adding mud data, loss/gain data and other data interpreted from drilling and testing history, plots these over the calculated data to help visually constrain the model

Option to include ascii file with observations from resistivity image logs (or others like ultrasonic scanning tools) to include tensile fractures and breakouts, which are then used to automatically constrain SHMax

##Installation:

You can download the setup file from the release section, and install it like any normal program.

Or, you can compile from source. To compile from source, you need toga and briefcase packages in your python environment.
'pip install toga'
'pip install briefcase'

Once these are installed, you can build using the briefcase commands
'briefcase create'
'briefcase build'
'briefcase package'

For more help, consult the beeware documentation.
