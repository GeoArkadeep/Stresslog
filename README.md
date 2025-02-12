
![BG1](https://github.com/GeoArkadeep/WellMasterGeoMech/assets/160126374/27efa304-817d-4520-a5f5-65e63e679c15)
# Stresslog

A package to calculate and plot pore pressure and other geomechanical data from las files. It is intended for pre-drill and post-drill studies of wells, and can be used for limited realtime stress prediction.

## Features

* Import las files, dlis files, deviation files, ucs and lithology files

* Alias logs using user modifiable alias file

* Pad log data all the way to surface, using appropriate KB, GL or WD values (Option to specify these and correct these in the Log Header)

* Calculate geomechanical properties using 6 component stress tensor 

* Calculate Pore Pressure, Shmin, SHMax, Sv and UCS and more

* Plot Stress Polygon, Wellbore Stability and more at drainhole analysis depth

* Plot the data in user configurable intervals

* Option to save Plots at custom DPI

* Export the well data as .csv, so you can run your own plots and calculations in your software of choice

* Output Las Files (with updated header and new data columns)

* Parameters for adding mud data, loss/gain data and other data interpreted from drilling and testing history, these are plotted over the calculated data to help visually constrain the model

* Option to include ascii file with observations from resistivity image logs (or others like ultrasonic calipers) to include tensile fractures and breakouts, which are then used internally to better constrain SHMax

* Override nu, mu, and UCS and tensile strength from the lithology file 

## Installation


````
pip install stresslog
````
Consult the documentation at https://stresslog.readthedocs.io/ for more help.

## Contributing

First things first: **Open source does not mean open contribution.** While this project is open source, we reserve the right to accept or reject contributions at our discretion. That said, we **welcome contributions**—feel free to submit a pull request!  

### Guidelines for Contributions  
✅ **Pull Requests are Welcome:** If you have an improvement, bug fix, or feature, send a PR.  
❌ **No Code-Style Requirements:** We don’t enforce a specific coding style—if the tests pass, your code is valid.  
👑 **Review is at Our Discretion:** We will review PRs **if we feel like it**. If we don’t, well… we won’t.  
🚫 **Rejections are Absolute:** If your PR gets the reply **"rejected by royal decree,"** that’s it—no further discussion.  
⚖ **Copyright Stays with Us:**  
   - **All forks and derivative works must retain the original copyright notices in every file.** This is a condition of using this codebase under AGPL Section 7.


## Code of Conduct  

This project follows a simple rule: **Don’t be evil.**  

- Be respectful when engaging with maintainers and other contributors.  
- Don't spam issues or PRs asking for approvals. We review things at our own pace.  
- **Forking this repo does not grant ownership.** All forks must comply with AGPL, and **the original copyright notices must be retained in all files**.  

That’s it. Play nice, write good code, and we’ll all get along fine. 🚀  



## Acknowledgements

Jon Jincai Zhang, for his awesome work on pore pressure prediction, and geomechanics at large. This software uses his algorithms wherever possible.

In more detail:
The Stability Analysis Plots are after [Peska and Zoback, 1995](https://doi.org/10.1029/95JB00319). The Stress Polygon Technique for constraining SHMax is after [Zoback et al., 2003](https://doi.org/10.1029/95JB00319) The Pore Pressure equation is from [Zhang, 2011](https://doi.org/10.1016/j.earscirev.2011.06.001) The shmin is calculated in accordance with [Dianes, 1982](https://doi.org/10.2118/9254-PA), as well as [Zoback and Healy,1992](https://doi.org/10.1029/91JB02175)

This software is written in Python, using the python library Welly by Agile Scientific, Canada. Welly itself uses the Lasio library to handle the las files. Dlis files are handled using dlisio package by Equinor. Other libraries used include Pandas, Numpy, Matplotlib, Plotly and Scipy.

## Disclaimer

IN MAKING INTERPRETATIONS OF LOGS THIS SOFTWARE AND ITS AUTHOR(S) WILL GIVE USERS THE BENEFIT OF THEIR BEST JUDGEMENT. BUT SINCE ALL INTERPRETATIONS ARE OPINIONS BASED ON INFERENCES FROM ELECTRICAL OR OTHER MEASUREMENTS, WE CANNOT, AND WE DO NOT GUARANTEE THE ACCURACY OR CORRECTNESS OF ANY INTERPRETATION. WE SHALL NOT BE LIABLE OR RESPONSIBLE FOR ANY LOSS, COST, DAMAGES, OR EXPENSES WHATSOEVER INCURRED OR SUSTAINED BY THE USER RESULTING FROM ANY INTERPRETATION MADE BY THE SOFTWARE OR ITS AUTHOR(S).

THERE IS NO WARRANTY FOR THE PROGRAM. THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

THE STATEMENTS ABOVE ARE IN ADDITION TO THE CONCERNED SECTIONS OF THE AGPL3.0 LICENSE GOVERNING THE PROGRAM.
