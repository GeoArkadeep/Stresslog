import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga import Window

import lasio as laua
import welly

import pandas as pd
import numpy as np

import functools
import os

   
import math
import threading
import asyncio
import concurrent.futures
from threading import Lock
model_lock = Lock()
import traceback
import pint
import json
import csv

import http.server
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import socketserver

from manage_preferences import show_preferences_window
from webedit import custom_edit

from geomechanics import plotPPzhang
from geomechanics import read_aliases_from_file, read_styles_from_file, read_pstyles_from_file, pad_val, find_nearest_depth, interpolate_nan, datasets_to_las

#Logging
import sys
import logging
from pathlib import Path

user_home = os.path.expanduser("~/Documents")
app_data = os.getenv("APPDATA")
output_dir = os.path.join(user_home, "Stresslog_Plots")
input_dir = os.path.join(user_home, "Stresslog_Models")
output_dir1 = os.path.join(user_home, "Stresslog_Data")
#motor_dir = os.path.join(user_home, "Mud_Motor")

os.makedirs(output_dir1, exist_ok=True)  # Ensure output_dir exists
# Set up logging
log_file = os.path.join(output_dir1, "Stresslog_log.txt")
#log_file = Path.home() / "Documents" / "Stresslog_Data" / "Stresslog_log.txt"
if os.path.isfile(log_file):
    try:
        os.remove(log_file)
        print(f"Previous log file deleted: {log_file}")
    except Exception as e:
        print(f"Error deleting previous log file: {e}")

logging.basicConfig(filename=str(log_file), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for all console output
console_logger = logging.getLogger('Console')

# Redirect stdout and stderr
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(console_logger, logging.INFO)
sys.stderr = StreamToLogger(console_logger, logging.ERROR)


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)
#os.makedirs(motor_dir, exist_ok=True)

output_file = os.path.join(output_dir, "PlotFigure.png")
output_fileS = os.path.join(output_dir, "PlotStability.png")
output_fileSP = os.path.join(output_dir, "PlotPolygon.png")
output_fileVec = os.path.join(output_dir, "PlotVec.png")
output_fileBHI = os.path.join(output_dir, "PlotBHI.png")
output_fileHoop = os.path.join(output_dir, "PlotHoop.png")
output_fileFrac = os.path.join(output_dir, "PlotFrac.png")
output_fileAll = os.path.join(output_dir, "PlotAll.png")
output_file2 = os.path.join(output_dir1, "output.csv")
output_forms = os.path.join(output_dir1, "tempForms.csv")
output_ucs = os.path.join(output_dir1, "tempUCS.csv")
output_lithology = os.path.join(output_dir1, "tempLitho.csv")
output_imagelog = os.path.join(output_dir1, "tempImage.csv")
output_file3 = os.path.join(output_dir1, "output.las")
modelpath = os.path.join(input_dir, "model.csv")
aliaspath = os.path.join(input_dir, "alias.txt")
unitpath = os.path.join(input_dir, "units.txt")
stylespath = os.path.join(input_dir, "styles.txt")
pstylespath = os.path.join(input_dir, "pstyles.txt")
#motor_db_path = os.path.join(motor_dir, "motor_db.json")
algopath = os.path.join(input_dir, "settings.txt")

path_dict = {}

# First define the base directories
path_dict['output_dir'] = os.path.join(user_home, "Stresslog_Plots")
path_dict['output_dir1'] = os.path.join(user_home, "Stresslog_Data")
path_dict['input_dir'] = os.path.join(user_home, "Stresslog_Models")
path_dict['motor_dir'] = os.path.join(user_home, "Mud_Motor")

# Then use them to define the file paths
path_dict.update({
    'plot_figure': os.path.join(path_dict['output_dir'], "PlotFigure.png"),
    'plot_stability': os.path.join(path_dict['output_dir'], "PlotStability.png"),
    'plot_polygon': os.path.join(path_dict['output_dir'], "PlotPolygon.png"),
    'plot_vec': os.path.join(path_dict['output_dir'], "PlotVec.png"),
    'plot_bhi': os.path.join(path_dict['output_dir'], "PlotBHI.png"),
    'plot_hoop': os.path.join(path_dict['output_dir'], "PlotHoop.png"),
    'plot_frac': os.path.join(path_dict['output_dir'], "PlotFrac.png"),
    'plot_all': os.path.join(path_dict['output_dir'], "PlotAll.png"),
    'output_csv': os.path.join(path_dict['output_dir1'], "output.csv"),
    'output_forms': os.path.join(path_dict['output_dir1'], "tempForms.csv"),
    'output_ucs': os.path.join(path_dict['output_dir1'], "tempUCS.csv"),
    'output_las': os.path.join(path_dict['output_dir1'], "output.las"),
    'model_path': os.path.join(path_dict['input_dir'], "model.csv"),
    'alias_path': os.path.join(path_dict['input_dir'], "alias.txt"),
    'unit_path': os.path.join(path_dict['input_dir'], "units.txt"),
    'styles_path': os.path.join(path_dict['input_dir'], "styles.txt"),
    'pstyles_path': os.path.join(path_dict['input_dir'], "pstyles.txt"),
    'motor_db_path': os.path.join(path_dict['motor_dir'], "motor_db.json")
})

os.remove(output_forms) if os.path.exists(output_forms) else None
os.remove(output_ucs) if os.path.exists(output_ucs) else None

import shutil
import requests

base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
files_to_check = {
   "plotly-2.34.0.min.js": "https://cdn.plot.ly/plotly-2.34.0.min.js",
   "BG1.png": "https://github.com/GeoArkadeep/Stresslog/raw/master/src/Stresslog/BG1.png",
   "BG2.png": "https://github.com/GeoArkadeep/Stresslog/raw/master/src/Stresslog/BG2.png"
}

def find_file(filename):
   for root, _, files in os.walk(base_dir):
       if filename in files:
           return os.path.join(root, filename)
   return None

for filename, url in files_to_check.items():
   output_file = os.path.join(output_dir, filename)
   if os.path.exists(output_file):
       continue
   local_file = find_file(filename)
   if local_file:
       shutil.copy(local_file, output_file)
       print(f"Copied {filename} to {output_dir}")
   else:
       try:
           response = requests.get(url)
           response.raise_for_status()
           with open(output_file, 'wb') as f:
               f.write(response.content)
           print(f"Downloaded {filename}")
       except requests.exceptions.RequestException as e:
           print(f"Failed to download {filename}: {e}")


class BackgroundImageView(toga.ImageView):
    def __init__(self, image_path, *args, **kwargs):
        super().__init__(image=toga.Image(image_path), *args, **kwargs)
        self.style.update(flex=1)

#Global Variables
laspath = None
devpath = None
lithopath = None
ucspath = None
flagpath = None
formpath = None
wella = None
#well2 =None
deva = None
lithos = None
UCSs = None
flags = None
forms=None
h1 = None
h2 = None
h3 = None
h4 = None
h5 = None
mwvalues = None
flowgradvals = None
fracgradvals= None
flowpsivals = None
fracpsivals = None
currentstatus = "Ready"
depth_track = None
finaldepth = None
attrib = [1,0,0,0,0,0,0,0]

prog_opts = [300, 0, 0, 0, 0]  # Default values

try:
    with open(algopath, 'r') as file:
        data = file.read().strip()  # Read the file and remove any trailing newline
        prog_opts = [int(float(num)) for num in data.split(',')]
except Exception:
    with open(algopath, 'w') as file:  # Open in write mode to overwrite with default values
        file.write(','.join(map(str, prog_opts)))

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
ureg.define('ppg = 0.051948 psi/foot')
ureg.define('sg = 0.4335 psi/foot = gcc = SG = GCC')
ureg.define('ksc = 1.0000005979/0.0703069999987293 psi = KSC = KSc = KsC = ksC = Ksc')
ureg.define('HR = hour')
ureg.define('M = meter')
ureg.define('mpa = MPa = Mpa')

try:
    with open(unitpath, 'r') as f:
        reader = csv.reader(f)
        unitchoice = next(reader)
        unitchoice = [int(x) for x in unitchoice]  # Convert strings to integers
        print(unitchoice)
except:
    unitchoice = [0,0,0,0,0] #Depth, pressure,gradient, strength, temperature

up = ['psi','Ksc','Bar','Atm','MPa']
us = ['MPa','psi','Ksc','Bar','Atm']
ug = ['gcc','sg','ppg','psi/foot']
ul = ['metres','feet']
ut = ['degC','degF','degR','degK']
unitdict = {"Depth": ["Metres", "Feet"],
            "Pressure": ['psi','Ksc','Bar','Atm','MPa'],
            "Gradient": ['G/CC','SG','PPG','psi/foot'],
            "Strength": ['MPa','psi','Ksc','Bar','Atm'],
            "Temperature": ["Centigrade", "Farenheit", "Rankine","Kelvin"]}

modelheader = "RhoA,AMC_exp,EATON_fac,tec_fac,NCT_exp,dtML,dtMAT,UL_exp,UL_depth,Res0,Be,Ne,Dex0,De,Nde,re_sub,A_dep,SHM_azi,Beta,Gamma,perm_cutoff,w_den,MudTempC,window,start,stop,nu_shale,nu_sst,nu_lst,dt_lst"
defaultmodel = "17,0.8,0.35,0,0.0008,250,60,0.0008,0,0.98,0.00014,0.6,0.5,0.00014,0.5,1,3500,0,0,0,0.35,1.025,60,21,0,2900,0.32,0.27,0.25,65"

print(os.getcwd())
try:
    data = pd.read_csv(modelpath,index_col=False)
except:
    file = open(modelpath,'w')
    file.write(modelheader+'\n')
    file.write(defaultmodel+'\n')
    file.close()
    data = pd.read_csv(modelpath,index_col=False)
# replacing end splitting the text  
# when newline ('\n') is seen. 
data_into_list = data.values.tolist()#(",") 
print(data_into_list) 
#modelfile.close() 

#model = np.array([16.33,0.63,0.0008,210,60,0.4,0.8,1,0,2000])
#model = ['16.33','0.63','0.0008','210','60','0.4','0.8','1','0','2000']
model = data_into_list[0]
print(model)
class MyApp(toga.App):
    global unitchoice
    def set_preferences(self, command):
        # This method sets up the preferences command
        self._preferences = command
    def preferences(self, widget):
        show_preferences_window(self, aliaspath, stylespath, pstylespath, unitdict, unitpath, algopath)
    def custom_edit_ucs(self, widget):
        self.run_custom_ucs()

    def run_custom_ucs(self):
        #global UCSs
        print(output_ucs)
        custom_edit(
            self, 
            ["MD", "UCS"], ["m","MPa"],output_ucs,8020
        )
        #print(UCSs)
        
    def custom_edit_forms(self, widget):
        self.run_custom_forms()

    def run_custom_forms(self):
        #global forms
        formunitdict = {"Depth": ["Metres", "Feet"],
            "None": [""]}
        custom_edit(
            self, 
            ["Top TVD", "Number", "Formation Name", "GR Cut", "Struc.Top", "Struc.Bottom", "CentroidRatio", "OWC", "GOC", "Coeff.Vol.Therm.Exp.","SHMax Azim.", "SVDip", "SVDipAzim","Tectonic Factor","InterpretedSH/Sh","Biot","Dt_NCT","Res_NCT","DXP_NCT"], ["m",""," ","","m","m","","m","m","","","","","","","","","",""],output_forms,8030
        )
        #print(forms)
    def custom_edit_imagelog(self, widget):
        self.run_custom_imagelog()

    def run_custom_imagelog(self):
        #global UCSs
        print(output_imagelog)
        custom_edit(
            self, 
            ["Top MD", "Observation"], ["m",""],output_imagelog,8040
        )
        #print(UCSs)
    def custom_edit_lithology(self, widget):
        self.run_custom_lithology()

    def run_custom_lithology(self):
        #global UCSs
        print(output_lithology)
        custom_edit(
            self, 
            ["Top MD", "Lithology Type","Interpreted Nu","Interpreted Mu","Interpreted UCS"], ["m","","","",""],output_lithology,8050
        )
        #print(UCSs)
        
    def startup(self):
        PREFERENCES = toga.Command(
            self.preferences,
            text='Preferences',
            shortcut=toga.Key.MOD_1 + 'p',
            tooltip = "Edit aliases, units and plot styles",
            group=toga.Group.FILE,
            #section=0
        )
        # Add the command to the app commands
        self.commands.add(PREFERENCES)
        
        load_las = toga.Command(
            self.open_las0,
            text='Load Las file',
            shortcut=toga.Key.MOD_1 + 'l',
            tooltip = "Load well log in Las format",
            group=toga.Group.FILE,
            #section=0
        )
        # Add the command to the app commands
        self.commands.add(load_las)
        
        load_dev = toga.Command(
            self.open_dev0,
            text='Load Deviation Survey',
            shortcut=toga.Key.MOD_1 + 'd',
            tooltip = "Load well survey ascii file",
            group=toga.Group.FILE,
            #section=0
        )
        # Add the command to the app commands
        self.commands.add(load_dev)
        
        load_ucs = toga.Command(
            self.open_ucs,
            text='Load core UCS data',
            tooltip = "Core UCS data for calibration",
            group=toga.Group.FILE,
            #section=0
        )
        # Add the command to the app commands
        self.commands.add(load_ucs)
        
        load_forms = toga.Command(
            self.open_formations,
            text='Load Formation Tops',
            #shortcut=toga.Key.MOD_1 + 'f',
            tooltip = "Formation Tops and properties",
            group=toga.Group.FILE,
            #section=0
        )
        
        # Add the command to the app commands
        self.commands.add(load_forms)
        
        load_lith = toga.Command(
            self.open_litho,
            text='Load Interpreted Lithology',
            #shortcut=toga.Key.MOD_1 + 'l',
            tooltip = "Interpreted Lithology in RockLab format",
            group=toga.Group.FILE,
            #section=0
        )
        # Add the command to the app commands
        self.commands.add(load_lith)
        load_image = toga.Command(
            self.open_flags,
            text='Load Interpreted Imagelog Observations',
            #shortcut=toga.Key.MOD_1 + 'l',
            tooltip = "Interpreted Imagelog Observations in RockLab format",
            group=toga.Group.FILE,
            #section=0
        )
        # Add the command to the app commands
        self.commands.add(load_image)

        # Explicitly set it as the preferences command
        #self.set_preferences(PREFERENCES)
        
        custom_edit_UCS = toga.Command(
            self.custom_edit_ucs,
            text='Edit UCS data',
            shortcut=toga.Key.MOD_1 + 'u',
            group=toga.Group.EDIT
        )
        self.commands.add(custom_edit_UCS)
        
        custom_edit_FORMS = toga.Command(
            self.custom_edit_forms,
            text='Edit Formation data',
            shortcut=toga.Key.MOD_1 + 'f',
            group=toga.Group.EDIT
        )
        self.commands.add(custom_edit_FORMS)
        
        custom_edit_FLAGS = toga.Command(
            self.custom_edit_imagelog,
            text='Edit Imagelog Observations',
            shortcut=toga.Key.MOD_1 + 'i',
            group=toga.Group.EDIT
        )
        self.commands.add(custom_edit_FLAGS)
        
        custom_edit_LITHO = toga.Command(
            self.custom_edit_lithology,
            text='Edit Lithology data',
            #shortcut=toga.Key.MOD_1 + 'f',
            group=toga.Group.EDIT
        )
        self.commands.add(custom_edit_LITHO)

        
        self.page1 = toga.Box(style=Pack(direction=COLUMN, flex=1))
        self.bg1 = BackgroundImageView("BG1.png", style=Pack(flex = 5))
        self.page1.add(self.bg1)

        spacer_box = toga.Box(style=Pack(flex=0.01))  # Add this spacer box
        self.page1.add(spacer_box)

        button_box1 = toga.Box(style=Pack(direction=ROW, alignment='center', flex=0))
        self.page1_btn1 = toga.Button("Load Las", on_press=self.open_las0, style=Pack(flex=1, padding=10))
        self.page1_btn2 = toga.Button("Load Dev txt", on_press=self.open_dev0, style=Pack(flex=1, padding=10), enabled = False)
        button_box1.add(self.page1_btn1)
        button_box1.add(self.page1_btn2)
        self.page1_btn3 = toga.Button("Well is Vertical", on_press=self.wellisvertical, style=Pack(flex=1, padding=10), enabled = False)
        self.page1_btn4 = toga.Button("Next", on_press=self.show_page2, style=Pack(flex=1, padding=10), enabled=False)
        button_box1.add(self.page1_btn3)
        button_box1.add(self.page1_btn4)

        self.page1.add(button_box1)
        #self.page1.add(button_box2)        
        self.dropdown1 = toga.Selection(style=Pack(padding=10), enabled = False)
        self.dropdown2 = toga.Selection(style=Pack(padding=10), enabled = False)
        self.dropdown3 = toga.Selection(style=Pack(padding=10), enabled = False)


          
        self.page2 = toga.Box(style=Pack(direction=COLUMN))
        self.page2_label = toga.Label("Measured Depth", style=Pack(padding=10))
        self.page2.add(self.page2_label)
        self.page2.add(self.dropdown1)
        self.page2_label = toga.Label("Inclination", style=Pack(padding=10))
        self.page2.add(self.page2_label)
        self.page2.add(self.dropdown2)
        self.page2_label = toga.Label("Azimuth", style=Pack(padding=10))
        self.page2.add(self.page2_label)
        self.page2.add(self.dropdown3)
        
        # Define the labels and default values
        entries_info2 = [
            {'label': 'KB', 'default_value': attrib[0]},
            {'label': 'GL', 'default_value': attrib[1]},
            {'label': 'TD', 'default_value': attrib[2]},
            {'label': 'Lat', 'default_value': attrib[3]},
            {'label': 'Long', 'default_value': attrib[4]},
            {'label': 'BHT', 'default_value': attrib[5]},
            {'label': 'Rm', 'default_value': attrib[6]},
            {'label': 'Rmf', 'default_value': attrib[7]}
        ]
        

        # Create a list to store the textboxes
        self.textboxes2 = []
        # Add 6 numeric entry boxes with their respective labels
        for i in range(2):
            entry_box2 = toga.Box(style=Pack(direction=ROW, alignment='center'))
            for j in range(4):
                entry_info2 = entries_info2[4*i+j]
                label = toga.Label(entry_info2['label'], style=Pack(padding_right=5, width=50, flex=1, text_direction='rtl'))
                entry2 = toga.TextInput(style=Pack(padding_left=2, flex=1))
                entry2.value = entry_info2['default_value']
                entry_box2.add(label)
                entry_box2.add(entry2)
                self.textboxes2.append(entry2)
            self.page2.add(entry_box2)
        
        
        # Step 1: Create a new Box to store the textboxes and buttons
        self.depth_mw_box = toga.Box(style=Pack(direction=COLUMN, alignment='center'))

        # Step 2: Initialize a list to store the textboxes' rows
        self.depth_mw_rows = []

        # Step 3: Create methods to add and remove rows
        def add_depth_mw_row(self, widget):
            row_box = toga.Box(style=Pack(direction=ROW, alignment='center', padding=5))
            
            depth_label = toga.Label("Casing Shoe Depth", style=Pack(padding_right=2,text_direction='rtl'))
            depth_entry = toga.TextInput(style=Pack(padding_left=5, flex=1), value="0")
            row_box.add(depth_label)
            row_box.add(depth_entry)

            mud_weight_label = toga.Label("Max. Mud Weight", style=Pack(padding_right=2,text_direction='rtl'))
            mud_weight_entry = toga.TextInput(style=Pack(padding_left=5, flex=1), value="1")
            row_box.add(mud_weight_label)
            row_box.add(mud_weight_entry)
            
            od_label = toga.Label("Casing OD (inches)", style=Pack(padding_right=2,text_direction='rtl'))
            od_entry = toga.TextInput(style=Pack(padding_left=5, flex=1), value="0")
            row_box.add(od_label)
            row_box.add(od_entry)
            
            bitdia_label = toga.Label("Bit Dia (inches)", style=Pack(padding_right=2,text_direction='rtl'))
            bitdia_entry = toga.TextInput(style=Pack(padding_left=5, flex=1), value="0")
            row_box.add(bitdia_label)
            row_box.add(bitdia_entry)
            
            iv_label = toga.Label("Mud Motor Identifier", style=Pack(padding_right=2,text_direction='rtl'))
            iv_entry = toga.TextInput(style=Pack(padding_left=5, flex=1), value="0")
            row_box.add(iv_label)
            row_box.add(iv_entry)
            
            ppf_label = toga.Label("BHT", style=Pack(padding_right=5,text_direction='rtl'))
            ppf_entry = toga.TextInput(style=Pack(padding_left=2, flex=1), value="0")
            row_box.add(ppf_label)
            row_box.add(ppf_entry)

            self.depth_mw_rows.append(row_box)
            self.depth_mw_box.add(row_box)

        def remove_depth_mw_row(self, widget):
            if len(self.depth_mw_rows) > 0:
                row_to_remove = self.depth_mw_rows.pop()
                self.depth_mw_box.remove(row_to_remove)

        # Step 4: Add the textboxes and buttons to the newly created box
        self.add_row_button = toga.Button("Add Row", on_press=lambda x: add_depth_mw_row(self, x), style=Pack(padding=5))
        self.remove_row_button = toga.Button("Remove Row", on_press=lambda x: remove_depth_mw_row(self, x), style=Pack(padding=5))

        button_box = toga.Box(style=Pack(direction=ROW, alignment='center', padding=5))
        button_box.add(self.add_row_button)
        button_box.add(self.remove_row_button)
        self.depth_mw_box.add(button_box)

        # Initialize the textboxes with 2 rows
        add_depth_mw_row(self, None)

        # Step 5: Add the new box to self.page2
        self.page2.add(self.depth_mw_box)
        
        self.page2_btn2 = toga.Button("Load Data and Proceed", on_press=self.show_page3, style=Pack(padding=10))
        self.page2.add(self.page2_btn2)
        
        self.page2_btn3 = toga.Button("Load Lithology from csv", on_press=self.open_litho, style=Pack(padding=10))
        self.page2.add(self.page2_btn3)
        
        self.page2_btn4 = toga.Button("Load UCS from csv", on_press=self.open_ucs, style=Pack(padding=10))
        self.page2.add(self.page2_btn4)
        
        self.page2_btn5 = toga.Button("Load Breakouts/DITFs from csv", on_press=self.open_flags, style=Pack(padding=10))
        self.page2.add(self.page2_btn5)
        
        self.page2_btn6 = toga.Button("Load Formations from csv", on_press=self.open_formations, style=Pack(padding=10))
        self.page2.add(self.page2_btn6)
        
        self.page2_btn1 = toga.Button("Back", on_press=self.show_page1, style=Pack(padding=10))
        self.page2.add(self.page2_btn1)
               
        
        #Page 3
        self.page3 = toga.Box(style=Pack(direction=ROW, alignment='center'))

        # Create a container with ROW direction for plot and frac_grad_data
        plot_and_data_box = toga.Box(style=Pack(direction=ROW, flex=1))

        # Initialize the flow_grad_data_box
        self.flow_grad_data_box = toga.Box(style=Pack(direction=COLUMN, flex=1))
        plot_and_data_box.add(self.flow_grad_data_box)

        # Add the buttons for Add row and Remove row
        flow_row_button_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
        self.add_flow_grad_button = toga.Button("Add PP Grad", on_press=lambda x: self.add_flow_grad_data_row(x, row_type='flow_grad'), style=Pack(flex=1))
        self.add_flow_psi_button = toga.Button("Add PP BHP", on_press=lambda x: self.add_flow_grad_data_row(x, row_type='flow_psi'), style=Pack(flex=1))
        flow_row_button_box.add(self.add_flow_grad_button)
        flow_row_button_box.add(self.add_flow_psi_button)
        self.flow_grad_data_box.add(flow_row_button_box)

        # Initialize the list of flow_grad_data rows
        self.flow_grad_data_rows = []
        self.add_flow_grad_data_row(None, row_type='flow_grad')
        self.add_flow_grad_data_row(None, row_type='flow_psi')

        
        #plot_and_data_box.add(self.webview1)

        # Initialize the frac_grad_data_box
        self.frac_grad_data_box = toga.Box(style=Pack(direction=COLUMN, flex=1))
        plot_and_data_box.add(self.frac_grad_data_box)

        # Add the buttons for Add row and Remove row
        row_button_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
        self.add_frac_grad_button = toga.Button("Add Frac Grad", on_press=lambda x: self.add_frac_grad_data_row(x, row_type='frac_grad'), style=Pack(flex=1))
        self.add_frac_psi_button = toga.Button("Add Frac BHP", on_press=lambda x: self.add_frac_grad_data_row(x, row_type='frac_psi'), style=Pack(flex=1))
        row_button_box.add(self.add_frac_grad_button)
        row_button_box.add(self.add_frac_psi_button)
        self.frac_grad_data_box.add(row_button_box)

        # Initialize the list of frac_grad_data rows
        self.frac_grad_data_rows = []
        self.add_frac_grad_data_row(None, row_type='frac_grad')
        self.add_frac_grad_data_row(None, row_type='frac_psi')

        # Create a scrollable container for the left pane
        left_pane_container = toga.ScrollContainer(style=Pack(width=300, padding=5))

        # Create the left pane box to hold all the elements in a vertical order
        left_pane_box = toga.Box(style=Pack(direction=COLUMN, alignment='center', flex=1))
        
        # Add the progress bar to the right pane box
        self.progress = toga.ProgressBar(max=None, style=Pack(alignment='center', height=5, flex=1))
        left_pane_box.add(self.progress)
        self.progress.stop()

        # Add the Recalculate button
        self.page3_btn1 = toga.Button("Recalculate", on_press=self.get_textbox_values, style=Pack(padding=5, flex=1))
        left_pane_box.add(self.page3_btn1)
        self.page3_btn5 = toga.Button("Stability Plot", on_press=self.show_page4, style=Pack(padding=5, flex=1), enabled=False)
        left_pane_box.add(self.page3_btn5)
        self.page3_btn4 = toga.Button("Back", on_press=self.show_page2, style=Pack(padding=5, flex=1))
        left_pane_box.add(self.page3_btn4)
        
        # Define the labels and default values
        global model
        entries_info = [
            {'label': 'RHOA (ppg)', 'default_value': str(model[0])},
            {'label': 'OBG Exp', 'default_value': str(model[1])},
            {'label': "Eaton's Nu", 'default_value': str(model[2])},
            {'label': 'TectonicFactor', 'default_value': str(model[3])},
            
            {'label': 'DT NCT Exponent', 'default_value': str(model[4])},
            {'label': 'DT @ mudline (us/ft)', 'default_value': str(model[5])},
            {'label': 'DT matrix (us/ft)', 'default_value': str(model[6])},
            {'label': 'Unloading Exp', 'default_value': str(model[7])},
            {'label': 'Unloading Depth', 'default_value': "0"},
            {'label': 'Resistivity @ mudline', 'default_value': str(model[9])},
            {'label': 'RES NCT Exponent', 'default_value': str(model[10])},
            {'label': 'RES PP Exponent', 'default_value': str(model[11])},
            {'label': 'D.Exponent @ mudline', 'default_value': str(model[12])},
            {'label': 'D.Exp NCT Exponent', 'default_value': str(model[13])},
            {'label': 'D.Exp PP Exponent', 'default_value': str(model[14])},
            {'label': 'PP Gr. L.Limit', 'default_value': str(model[15])},

            {'label': 'Analysis TVD', 'default_value': "0"},
            {'label': 'Fast Shear Azimuth', 'default_value': str(model[17])},
            {'label': 'Dip Azim.', 'default_value': str(model[18])},
            {'label': 'Dip Angle', 'default_value': str(model[19])},
            
            {'label': 'ShaleFlag Cutoff', 'default_value': str(model[20])},
            {'label': 'WaterDensity', 'default_value': str(model[21])},
            {'label': 'MudTemp', 'default_value': str(model[22])},
            
            {'label': 'Window', 'default_value': str(model[23])},
            {'label': 'Start', 'default_value': str(model[24])},
            {'label': 'Stop', 'default_value': str(model[25])}
            
            
            
        ]
        
        # Create a list to store the textboxes
        self.textboxes = []

        # Function to create a divider
        def create_divider(text):
            divider_box = toga.Box(style=Pack(direction=ROW, padding=(10, 5)))
            divider_label = toga.Label(text, style=Pack(flex=1, font_weight='bold'))
            divider_box.add(divider_label)
            return divider_box

        # Function to create a parameter row
        def create_parameter_row(entry_info):
            row_box = toga.Box(style=Pack(direction=ROW, alignment='left', padding=5))
            label = toga.Label(entry_info['label'], style=Pack(flex=1, padding=(5, 0)))
            entry = toga.TextInput(style=Pack(flex=1, padding=(.5, 0)))
            entry.value = entry_info['default_value']
            row_box.add(label)
            row_box.add(entry)
            return row_box, entry

        # Replace the existing loop with this code
        current_index = 0

        # Frac Grad Properties
        left_pane_box.add(create_divider("Frac Grad Parameters"))
        for i in range(4):
            row_box, entry = create_parameter_row(entries_info[current_index])
            left_pane_box.add(row_box)
            self.textboxes.append(entry)
            current_index += 1

        # Pore Pressure Properties
        left_pane_box.add(create_divider("Pore Pressure Parameters"))
        for i in range(12):
            row_box, entry = create_parameter_row(entries_info[current_index])
            left_pane_box.add(row_box)
            self.textboxes.append(entry)
            current_index += 1

        # Misc. Properties
        left_pane_box.add(create_divider("Stress Tensor Parameters"))
        for i in range(4):
            row_box, entry = create_parameter_row(entries_info[current_index])
            left_pane_box.add(row_box)
            self.textboxes.append(entry)
            current_index += 1

        # Display Properties
        left_pane_box.add(create_divider("Misc. Properties"))
        for i in range(3):
            row_box, entry = create_parameter_row(entries_info[current_index])
            left_pane_box.add(row_box)
            self.textboxes.append(entry)
            current_index += 1

        # Stress Tensor Properties
        left_pane_box.add(create_divider("Display Properties"))
        for i in range(3):
            row_box, entry = create_parameter_row(entries_info[current_index])
            left_pane_box.add(row_box)
            self.textboxes.append(entry)
            current_index += 1
        
        left_pane_box.add(create_divider("Constraints"))

        # Add the containers for added rows to the left pane box
        left_pane_box.add(self.flow_grad_data_box)
        left_pane_box.add(self.frac_grad_data_box)

        self.page3_btn2 = toga.Button("Export Plot", on_press=self.save_plot, style=Pack(padding=5, flex=1))
        left_pane_box.add(self.page3_btn2)
        self.page3_btn3 = toga.Button("Export Las", on_press=self.save_las, style=Pack(padding=5, flex=1))
        left_pane_box.add(self.page3_btn3)       

        # Add the left pane box to the scrollable container
        left_pane_container.content = left_pane_box
        
        # Create a scrollable container for the right pane
        right_pane_container = toga.Box(style=Pack(direction='row',flex=1))

        # Create a box for the progress bar and image
        #right_pane_box = toga.Box(style=Pack(direction=ROW, flex=1))
        
        # Adjust the ScrollContainer to handle overflow properly
        #right_pane_container.content = right_pane_box
        #right_pane_container.horizontal = False
        girth = 1000

        # Add the image display to the right pane box
        my_image = toga.Image("BG2.png")
        self.start_server2()
        self.bg_img_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no, maximum-scale=1.0">
    <title>Dynamic Background</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            overflow: hidden; /* Prevent scrolling */
            height: 100vh; /* Full viewport height */
            width: 100vw; /* Full viewport width */
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: transparent; /* Transparent background */
        }

        .container {
            position: relative;
            height: 100%; 
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: transparent; /* Optional for visibility */
        }

        img {
            width: 100%; /* Take full width */
            height: auto; /* Maintain aspect ratio */
            max-height: 100%; /* Ensure it doesn't overflow vertically */
            object-fit: cover; /* Crop vertically if needed */
            object-position: center; /* Center the image vertically */
        }
    </style>
</head>
<body>
    <div class="container">
        <img src="http://localhost:8010/BG2.png" alt="Dynamic Image">
    </div>
    
    <script>
        // Prevent zooming with keyboard shortcuts
        window.addEventListener('keydown', function (event) {
            if ((event.ctrlKey || event.metaKey) && (event.key === '+' || event.key === '-' || event.key === '0')) {
                event.preventDefault();
            }
        });

        // Prevent pinch zoom on touch devices
        window.addEventListener('wheel', function (event) {
            if (event.ctrlKey) {
                event.preventDefault();
            }
        }, { passive: false });

        // Prevent double-tap zoom
        let lastTouchEnd = 0;
        document.addEventListener('touchend', function (event) {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
    </script>
</body>
</html>
"""
        print("starting webview background")
        self.webview1 = toga.WebView(style=Pack(flex=1))
        self.webview1.set_content(content=self.bg_img_html, root_url="http://localhost:8010/")
        #right_pane_container.add(self.bg3)
        right_pane_container.add(self.webview1)

        # Add the containers to the main page3 box
        self.page3.add(left_pane_container)
        self.page3.add(right_pane_container)        
        
        
        #Page4
        
        self.page4 = toga.Box(style=Pack(direction=COLUMN, alignment='center'))
        self.dbox = toga.Box(style=Pack(direction=ROW, alignment='center',flex=1))
        #self.dbox2 = toga.Box(style=Pack(direction=ROW, alignment='center',flex=1))
        #self.bg5 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        #self.dbox.add(self.bg5)

        self.bg4 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        self.page4.add(self.bg4)
        
        #self.bg6 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        #self.dbox2.add(self.bg6)

        #self.bg7 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        #self.dbox2.add(self.bg7)
        
        button_box4 = toga.Box(style=Pack(direction=ROW, alignment='center', flex=0))
        
        self.page4_btn1 = toga.Button("Start Over", on_press=self.show_page1, style=Pack(padding=1))
        button_box4.add(self.page4_btn1)
        
        self.page4_btn2 = toga.Button("Export Plot", on_press=self.save_las, style=Pack(padding=1))
        button_box4.add(self.page4_btn2)
        
        self.page4_btn3 = toga.Button("Back", on_press=self.show_page3, style=Pack(padding=1))
        button_box4.add(self.page4_btn3)
        
        #self.page4.add(self.dbox)
        #self.page4.add(self.dbox2)
        self.page4.add(button_box4)
        
        self.main_window = toga.MainWindow(title=self.formal_name,size=[1080,720])
        self.main_window.content = self.page1
        self.main_window.show()
        

        
    def add_frac_grad_data_row(self, widget, row_type='frac_grad'):
        depth_label = toga.Label("MD", style=Pack(text_align="center", flex=1, padding_top=5))
        
        if row_type == 'frac_grad':
            second_label = toga.Label("Frac Grad "+ug[unitchoice[2]], style=Pack(text_align="center", flex=1, padding_top=5))
        elif row_type == 'frac_psi':
            second_label = toga.Label("Frac BHP "+up[unitchoice[1]], style=Pack(text_align="center", flex=1, padding_top=5))
        else:
            raise ValueError("Invalid row type")

        depth_input = toga.TextInput(style=Pack(flex=1), value="0")
        second_input = toga.TextInput(style=Pack(flex=1), value="0")

        row_labels = toga.Box(style=Pack(direction=ROW, padding_top=5))
        row_labels.add(depth_label)
        row_labels.add(second_label)

        row_inputs = toga.Box(style=Pack(direction=ROW, padding_top=5))
        row_inputs.add(depth_input)
        row_inputs.add(second_input)

        self.frac_grad_data_box.add(row_labels)
        self.frac_grad_data_box.add(row_inputs)

        self.frac_grad_data_rows.append((depth_input, second_input, row_type))

    def remove_frac_grad_data_row(self, widget):
        if len(self.frac_grad_data_rows) > 0:
            row = self.frac_grad_data_rows.pop()
            self.frac_grad_data_box.remove(row[2])
            
    def add_flow_grad_data_row(self, widget, row_type='flow_grad'):
        depth_label = toga.Label("MD", style=Pack(text_align="center", flex=1, padding_top=5))
        
        if row_type == 'flow_grad':
            second_label = toga.Label("PP Grad "+ug[unitchoice[2]], style=Pack(text_align="center", flex=1, padding_top=5))
        elif row_type == 'flow_psi':
            second_label = toga.Label("PP BHP "+up[unitchoice[1]], style=Pack(text_align="center", flex=1, padding_top=5))
        else:
            raise ValueError("Invalid row type")

        depth_input = toga.TextInput(style=Pack(flex=1), value="0")
        second_input = toga.TextInput(style=Pack(flex=1), value="0")

        row_labels = toga.Box(style=Pack(direction=ROW, padding_top=5))
        row_labels.add(depth_label)
        row_labels.add(second_label)

        row_inputs = toga.Box(style=Pack(direction=ROW, padding_top=5))
        row_inputs.add(depth_input)
        row_inputs.add(second_input)

        self.flow_grad_data_box.add(row_labels)
        self.flow_grad_data_box.add(row_inputs)

        self.flow_grad_data_rows.append((depth_input, second_input, row_type))
    
    def remove_flow_grad_data_row(self, widget):
        if len(self.flow_grad_data_rows) > 1:
            row_to_remove = self.flow_grad_data_rows.pop()
            self.flow_grad_data_box.remove(row_to_remove)
            
    
    def show_page1(self, widget):
        global flagpath,formpath,ucspath,lithopath,devpath,laspath,h1
        devpath = None
        laspath = None
        ucspath = None
        lithopath = None
        flagpath = None
        formpath = None
        h1 = None
        self.dropdown1.items = []
        self.dropdown2.items = []
        self.dropdown3.items = []
        self.page1_btn4.enabled = False
        self.page1_btn3.enabled = False
        self.page1_btn2.enabled = False
        self.dropdown1.enabled = False
        self.dropdown2.enabled = False
        self.dropdown3.enabled = False
        self.main_window.content = self.page1

    def show_page2(self, widget):
        self.main_window.content = self.page2
    
    def show_page3(self, widget):
        self.set_textbox2_values(widget)
        self.main_window.content = self.page3
    
    def show_page4(self, widget):
        self.main_window.content = self.page4


    def get_frac_grad_data_values(self):
        frac_grad_values = []
        frac_psi_values = []

        for depth_input, second_input, row_type in self.frac_grad_data_rows:
            try:
                depth = float(depth_input.value)
                second_value = float(second_input.value)
            except ValueError:
                print("Invalid input. Skipping this row.")
                continue

            if row_type == 'frac_grad':
                frac_grad_values.append([(float(second_value)*ureg(ug[unitchoice[2]])).to('gcc').magnitude,(float(depth)*ureg(ul[unitchoice[0]])).to('metre').magnitude])
            elif row_type == 'frac_psi':
                frac_psi_values.append([(float(second_value)*ureg(up[unitchoice[1]])).to('psi').magnitude,(float(depth)*ureg(ul[unitchoice[0]])).to('metre').magnitude])

        frac_grad_values.sort(key=lambda x: x[0])
        frac_psi_values.sort(key=lambda x: x[0])

        return frac_grad_values, frac_psi_values

    def get_flow_grad_data_values(self):
        flow_grad_values = []
        flow_psi_values = []

        for depth_input, second_input, row_type in self.flow_grad_data_rows:
            try:
                depth = float(depth_input.value)
                second_value = float(second_input.value)
            except ValueError:
                print("Invalid input. Skipping this row.")
                continue

            if row_type == 'flow_grad':
                flow_grad_values.append([(float(second_value)*ureg(ug[unitchoice[2]])).to('gcc').magnitude, (float(depth)*ureg(ul[unitchoice[0]])).to('metre').magnitude])
            elif row_type == 'flow_psi':
                flow_psi_values.append([(float(second_value)*ureg(up[unitchoice[1]])).to('psi').magnitude, (float(depth)*ureg(ul[unitchoice[0]])).to('metre').magnitude])

        flow_grad_values.sort(key=lambda x: x[0])
        flow_psi_values.sort(key=lambda x: x[0])

        return flow_grad_values, flow_psi_values
    
    def get_depth_mw_data_values(self):
        depth_mw_values = []

        for row_box in self.depth_mw_rows:
            depth_entry = row_box.children[1] # Access the depth TextInput widget
            mw_entry = row_box.children[3] # Access the mud weight TextInput widget
            od_entry = row_box.children[5] # Access the mud weight TextInput widget
            bd_entry = row_box.children[7] # Access the mud weight TextInput widget
            iv_entry = row_box.children[9] # Access the mud weight TextInput widget
            bht_entry = row_box.children[11] #access the casing volume TextInput Widget

            try:
                depth = (float(depth_entry.value)*ureg(ul[unitchoice[0]])).to('metre').magnitude
                mw = (float(mw_entry.value)*ureg(ug[unitchoice[2]])).to('gcc').magnitude
                od = float(od_entry.value)
                bd = float(bd_entry.value)
                iv = iv_entry.value
                sec_bht = (float(bht_entry.value)*ureg(ut[unitchoice[4]])).to('degC').magnitude if float(bht_entry.value) !=0 else 0
            except ValueError:
                print("Invalid input. Skipping this row.")
                continue

            depth_mw_values.append([mw,depth,bd,od,iv,sec_bht])

        # Sort the depth_mw_values list by depth
        depth_mw_values.sort(key=lambda x: x[1])

        return depth_mw_values
    
    def getwelldev(self):
        global laspath, devpath, wella, deva, depth_track, finaldepth
        print(self.dropdown1.value)
        print("Recalculating....   "+str(laspath))
        wella = welly.Well.from_las(laspath, index = "m")
        depth_track = wella.df().index
        if devpath is not None:
            deva=pd.read_csv(devpath, sep=r'[ ,	]',skipinitialspace=True)
            print(deva)
            start_depth = wella.df().index[0]
            final_depth = wella.df().index[-1]
            spacing = ((wella.df().index.values[-1]-wella.df().index.values[0])/len(wella.df().index.values))
            print("Sample interval is :",spacing)
            padlength = int(start_depth/spacing)
            print(padlength)
            padval = np.zeros(padlength)
            i = 1
            while(i<padlength):
                padval[-i] = start_depth-(spacing*i)
                i+=1
            print("pad depths: ",padval)
            md = depth_track
            md =  np.append(padval,md)
            mda = pd.to_numeric(deva[self.dropdown1.value], errors='coerce')
            inca = pd.to_numeric(deva[self.dropdown2.value], errors='coerce')
            azma = pd.to_numeric(deva[self.dropdown3.value], errors='coerce')
            inc = np.interp(md,mda,inca)
            azm = np.interp(md,mda,azma)
            #i = 1
            #while md[i]<final_depth:
            #    if md[i]
            z = deva.to_numpy(na_value=0)
            dz = [md,inc,azm]
        else:
            start_depth = wella.df().index[0]
            final_depth = wella.df().index[-1]
            spacing = ((wella.df().index.values[-1]-wella.df().index.values[0])/len(wella.df().index.values))
            print("Sample interval is :",spacing)
            padlength = int(start_depth/spacing)
            print(padlength)
            padval = np.zeros(padlength)
            i = 1
            while(i<padlength):
                padval[-i] = start_depth-(spacing*i)
                i+=1
            print("pad depths: ",padval)
            
            md = depth_track
            md =  np.append(padval,md)
            #md[0] = 0
            #md[0:padlength-1] = padval[0:padlength-1]
            inc = np.zeros(len(depth_track)+padlength)
            azm = np.zeros(len(depth_track)+padlength)
            dz = [md,inc,azm]



        dz = np.transpose(dz)
        dz = pd.DataFrame(dz)
        dz = dz.dropna()
        print(dz)
        finaldepth = dz.to_numpy()[-1][0]
        print("Final depth is ",finaldepth)
        wella.location.add_deviation(dz, wella.location.td)
        tvdg = wella.location.tvd
        md = wella.location.md
        from welly import curve
        MD = curve.Curve(md, mnemonic='MD',units='m', index = md)
        wella.data['MD'] =  MD
        TVDM = curve.Curve(tvdg, mnemonic='TVDM',units='m', index = md)
        wella.data['TVDM'] =  TVDM
        wella.unify_basis(keys=None, alias=None, step=spacing)
        
        #self.bg3.image = toga.Image('BG1.png')
        #smoothass.plotPPzhang(wella)
        print("Great Success!! :D")
        #image_path = 'PlotFigure.png'
        #self.bg3.image = toga.Image(image_path)


    def on_result0(self, widget):
        global wella, laspath
        if laspath is not None:
            wella = welly.Well.from_las(laspath, index = "m")
            print(wella)
            print(wella.header)
            choptop = 200
            for curve_name in wella.data.keys():
                curve = wella.data[curve_name]
                mask = curve.basis > (curve.start + choptop)
                wella.data[curve_name] = curve.to_basis(curve.basis[mask])
            self.get_textbox2_values(widget)

            self.page1_btn2.enabled = True
            self.page1_btn3.enabled = True
            #return wella
        else:
            print("No file selected.")


    async def open_las0(self, widget):
        global laspath
        try:
            laspath_dialog = await self.main_window.open_file_dialog(title="Select a las file",file_types=['las'], multiple_select=False)
            if laspath_dialog:  # Check if the user selected a file and didn't cancel the dialog
                laspath = laspath_dialog
                self.on_result0(widget)
            else:
                print("File selection was canceled.")
        except Exception as e:
            print("Error:", e)

    def on_result1(self, widget):
        global h1, devpath
        if devpath is not None:               
            h1 = readDevFromAsciiHeader(devpath)
            print("Loaded dev file:", devpath)
            print(h1)
            self.populate_dropdowns()
            self.dropdown1.enabled = True
            self.dropdown2.enabled = True
            self.dropdown3.enabled = True
            self.page1_btn4.enabled = True
            self.page1_btn3.enabled = False
            try:
                self.getwelldev()
            except:
                print("Load the damn well log first, fool!")
                pass
        else:
            print(wella)



    async def open_dev0(self, widget):
        global devpath
        try:
            devpath = await self.main_window.open_file_dialog(title="Select a Dev file", multiselect=False)
            self.on_result1(widget)
        except Exception as e:
            print("Error:", e)
            
    def on_result2(self, widget):
        global h2, lithopath
        if lithopath is not None:               
            h2 = readLithoFromAscii(lithopath)
            print("Loaded litho file:", lithopath)
            print(h2)
            h2.to_csv(output_lithology)
            self.run_custom_lithology()
            
        else:
            print("No litho file loaded")

    def on_result3(self, widget):
        global h3, ucspath
        if ucspath is not None:               
            h3 = readUCSFromAscii(ucspath)
            print("Loaded ucs file:", ucspath)
            print(h3)
            h3.to_csv(output_ucs)
            self.run_custom_ucs()
            
            
        else:
            print("No ucs file loaded")
    
    def on_result4(self, widget):
        global h4, flagpath
        if flagpath is not None:               
            h4 = readFlagFromAscii(flagpath)
            print("Loaded flag file:", flagpath)
            print(h4)           
            h4.to_csv(output_imagelog)
            self.run_custom_imagelog()
        else:
            print("No flag file loaded")

    def on_result5(self, widget):
        global h5, formpath
        if formpath is not None:               
            h5 = readFormFromAscii(formpath)
            print("Loaded formation file:", formpath)
            print(h5)
            h5.to_csv(output_forms)
            self.run_custom_forms()
            
        else:
            print("No formation file loaded")

    async def open_litho(self, widget):
        global lithopath
        try:
            lithopath = await self.main_window.open_file_dialog(title="Select a Litho file", multiselect=False)
            self.on_result2(widget)
        except Exception as e:
            print("Error:", e)
            
    async def open_ucs(self, widget):
        global ucspath
        try:
            ucspath = await self.main_window.open_file_dialog(title="Select a UCS file", multiselect=False)
            self.on_result3(widget)
        except Exception as e:
            print("Error:", e)
            
    async def open_flags(self, widget):
        global flagpath
        try:
            flagpath = await self.main_window.open_file_dialog(title="Select a Flag file", multiselect=False)
            self.on_result4(widget)
        except Exception as e:
            print("Error:", e)
    
    async def open_formations(self, widget):
        global formpath
        try:
            formpath = await self.main_window.open_file_dialog(title="Select a Formation file", multiselect=False)
            self.on_result5(widget)
        except Exception as e:
            print("Error:", e)
    
    def save_las(self,widget):
        #global wella
        #well = wella
        #print(well)
        name = wella.name
        name = name.translate({ord(i): '_' for i in '/\:*?"<>|'})
        output_file4 = os.path.join(output_dir1,name+"_GMech.las")
        df3 = wella.df()
        df3.index.name = 'DEPT'
        df3 = df3.reset_index()
        lasheader = wella.header
        print(lasheader,df3)
        c_units = {"TVDM":"M","RHO":"G/C3", "OBG_AMOCO":"G/C3", "DTCT":"US/F", "PP_DT_Zhang":"G/C3","FG_DAINES":"G/C3","GEOPRESSURE":"PSI","FRACTURE_PRESSURE":"PSI", "SHMAX_PRESSURE":"PSI", "shmin_PRESSURE":"PSI","MUD_PRESSURE":"PSI", "MUD_GRADIENT":"G/C3", "UCS_Horsud":"MPA", "UCS_Lal":"MPA"}
        datasets_to_las(output_file4, {'Header': lasheader,'Curves':df3}, c_units)
        global devpath,laspath
        devpath=None
        laspath=None
        #well2 = wella.from_df(df3)
        #wella.to_las(output_file4)
        self.show_page1(widget)
        
    def save_plot(self,widget):
        #global wella
        #well = wella
        #print(well)
        name = wella.name
        name = name.translate({ord(i): '_' for i in '/\:*?"<>|'})
        output_filePNG = os.path.join(output_dir,name+"_GMech.png")
        plt.savefig(output_filePNG,dpi=1200)
        plt.close()
        self.show_page1(widget)
        
        
    def populate_dropdowns(self):
        global h1
        self.dropdown1.items = h1
        self.dropdown2.items = h1[1:] + h1[:1]
        self.dropdown3.items = h1[2:] + h1[:2]
    
    def get_textbox2_values(self,widget):
        global wella
        global attrib
        try:
            attrib[0] = wella.location.ekb
        except AttributeError:
            pass
        try:
            attrib[1] = wella.location.egl
        except AttributeError:
            pass
        try:
            attrib[2] = wella.location.tdl
        except AttributeError:
            pass
        try:
            attrib[3] = wella.location.latitude
        except AttributeError:
            pass
        try:
            attrib[4] = wella.location.longitude
        except AttributeError:
            pass
        print(attrib)
        i=0
        for textbox in self.textboxes2:
            textbox.value = attrib[i]
            i+=1
        
    def set_textbox2_values(self,widget):
        global wella
        global attrib
        tv = [textbox.value for textbox in self.textboxes2]
        tv[0] = (float(tv[0])*ureg(ul[unitchoice[0]])).to('metre').magnitude
        tv[1] = (float(tv[1])*ureg(ul[unitchoice[0]])).to('metre').magnitude
        wella.location.ekb = tv[0]
        wella.location.kb = tv[0]
        wella.location.egl = tv[1]
        wella.location.gl = tv[1]
        #wella.location.tdl = tv[2]
        #wella.location.td = tv[2]
        wella.location.latitude = tv[3]
        wella.location.longitude = tv[4]
        tv[5] = (float(tv[5])*ureg(ut[unitchoice[4]])).to('degC').magnitude if float(tv[5]) !=0 else 0
        wella.header.bht = tv[5]
        wella.header.rm = tv[6]
        wella.header.rmf = tv[7]
        attrib=tv
        print('Attributes set according to following:')
        print(attrib)
        print(wella.location.egl)
        #wella.unify_basis(keys=None, alias=None, basis=md)
        
    def plotPPzhang_wrapper(self,well, kwargs):
        plotPPzhang(
            well,
            kwargs['rhoappg'],
            kwargs['lamb'],
            kwargs['ul_exp'],
            kwargs['ul_depth'],
            kwargs['a'],
            kwargs['nu'],
            kwargs['sfs'],
            kwargs['window'],
            kwargs['zulu'],
            kwargs['tango'],
            kwargs['dtml'],
            kwargs['dtmt'],
            kwargs['water'],
            kwargs['underbalancereject'],
            kwargs['tecb'],
            kwargs['doi'],
            kwargs['offset'],
            kwargs['strike'],
            kwargs['dip'],
            kwargs['mudtemp'],
            kwargs['res0'],
            kwargs['be'],
            kwargs['ne'],
            kwargs['dex0'],
            kwargs['de'],
            kwargs['nde'],
            kwargs['lala'],
            kwargs['lalb'],
            kwargs['lalm'],
            kwargs['lale'],
            kwargs['lall'],
            kwargs['horsuda'],
            kwargs['horsude'],
            kwargs['unitchoice'],
            kwargs['ureg'],
            kwargs['mwvalues'],
            kwargs['flowgradvals'],
            kwargs['fracgradvals'],
            kwargs['flowpsivals'],
            kwargs['fracpsivals'],
            kwargs['attrib'],
            kwargs['flags'],
            kwargs['UCSs'],
            kwargs['forms'],
            kwargs['lithos'],
            kwargs['user_home'],
            kwargs['paths'],
            kwargs['program_option']
            
        )
    
    def plotppwrapper(self, loop, *args, **kwargs):
        print("thread spawn")
        try:
            result = self.plotPPzhang_wrapper(*args, **kwargs)
            asyncio.run_coroutine_threadsafe(self.onplotfinish(), loop)
        except Exception as e:
            error_message = str(e)
            traceback_str = traceback.format_exc()
            print(f"Error in thread: {error_message}")
            print(f"Traceback: {traceback_str}")
            asyncio.run_coroutine_threadsafe(self.show_error_dialog(error_message), loop)
            asyncio.run_coroutine_threadsafe(self.onplotfuckup(traceback_str), loop)
            
        print("Thread despawn")
        return
    
    def start_plotPPzhang_thread(self, loop, *args, **kwargs):
        thread = threading.Thread(target=self.plotppwrapper, args=(loop, *args), kwargs=kwargs)
        thread.start()
        return
    
    async def show_error_dialog(self, error_message):
        # This method should be called on the main thread
        self.main_window.error_dialog('Error:', error_message)
        
    async def onplotfuckup(self,error_message):
        #self.start_server2()
        self.page3_btn1.enabled = True
        self.page3_btn2.enabled = True
        self.page3_btn3.enabled = True
        self.page3_btn4.enabled = True
        self.progress.stop()
        if float(model[16]) > 0:
            self.page3_btn5.enabled = True
            self.bg4.image = toga.Image(output_fileAll)
        else:
            self.page3_btn5.enabled = False
        self.start_server2()
        self.img_html = f"""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #ffffff;
            color: #000000;
            text-align: center;
            padding: 50px;
        }}
        .error-box {{
            border: 1px solid #ffffff;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            display: inline-block;
        }}
        h1 {{
            color: #000000;
        }}
        p {{
            margin: 10px 0;
        }}
        .try-again {{
            color: #155724;
            background-color: #d4edda;
            padding: 10px;
            border-radius: 5px;
            display: inline-block;
        }}
        a {{
            color: #007bff;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .file-path {{
            font-weight: bold;
            color: #d9534f;
        }}
    </style>
</head>
<body>
    <div class="error-box">
        <h1>Wow, that did not go as planned!</h1>
        <p>{error_message}</p>
        <h1>Please check all input and try again.</h1>
        <div class="try-again">
            If the traceback above did not help you understand what went wrong, 
            please email RockLab support at 
            <a href="mailto:support@rocklab.in?subject=Issue with Stresslog&body=Please attach the log file from the path below:">support@rocklab.in</a> 
            and include the log file found at the following location:
            <br><br>
            <span class="file-path">{log_file}</span>
        </div>
    </div>
</body>
</html>

"""
        self.webview1.set_content(content=self.img_html, root_url="http://localhost:8010/")
        print("Wrapper done")
        return
    
    async def onplotfinish(self):
        #self.start_server2()
        self.page3_btn1.enabled = True
        self.page3_btn2.enabled = True
        self.page3_btn3.enabled = True
        self.page3_btn4.enabled = True
        self.progress.stop()
        if float(model[16]) > 0:
            self.page3_btn5.enabled = True
            self.bg4.image = toga.Image(output_fileAll)
        else:
            self.page3_btn5.enabled = False
        self.start_server2()
        self.img_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <meta name="HandheldFriendly" content="true">
    <title>Plotly Charts with fetchWithRetry</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            overflow: hidden;
            height: 100%;
            width: 100%;
            /* Disable text selection */
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            /* Prevent iOS text size adjust on orientation change */
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
            /* Disable overscroll behavior */
            overscroll-behavior: none;
        }
        #top-plotly-chart, #main-plotly-chart {
            width: 100%;
            overflow: hidden;
        }
        #top-plotly-chart {
            position: fixed;
            top: 0;
            left: 0;
            height: 20%;
            z-index: 1000;
        }
        #main-plotly-chart {
            position: fixed;
            top: 20%;
            left: 0;
            height: 80%;
        }
        #divider {
            position: fixed;
            top: 20%;
            left: 0;
            width: 100%;
            height: 1px;
            background-color: black;
            z-index: 1001;
        }
    </style>
</head>
<body>
    <div id="top-plotly-chart"></div>
    <div id="divider"></div>
    <div id="main-plotly-chart"></div>
    <script>
        console.log('Script execution started');

        // Disable browser back functionality
        history.pushState(null, null, document.URL);
        window.addEventListener('popstate', function () {
            history.pushState(null, null, document.URL);
        });

        // Prevent zooming
        function preventZoom(e) {
            var t2 = e.timeStamp;
            var t1 = e.currentTarget.dataset.lastTouch || t2;
            var dt = t2 - t1;
            var fingers = e.touches.length;
            e.currentTarget.dataset.lastTouch = t2;

            if (!dt || dt > 500 || fingers > 1) return; // not double-tap

            e.preventDefault();
            e.target.click();
        }

        document.addEventListener('touchstart', preventZoom, {passive: false});

        // Prevent pinch zooming
        document.addEventListener('touchmove', function (e) {
            if (e.scale !== 1) {
                e.preventDefault();
            }
        }, { passive: false });

        // Prevent zoom on double tap
        var lastTouchEnd = 0;
        document.addEventListener('touchend', function (e) {
            var now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                e.preventDefault();
            }
            lastTouchEnd = now;
        }, false);

        // Prevent zoom on mouse wheel
        document.addEventListener('wheel', function(e) {
            if(e.ctrlKey) {
                e.preventDefault();
            }
        }, { passive: false });

        // Prevent zoom on keydown (Ctrl + '+' or Ctrl + '-')
        document.addEventListener('keydown', function(e) {
            if(e.ctrlKey && (e.key === '+' || e.key === '-' || e.key === '=')) {
                e.preventDefault();
            }
        }, false);

        // Prevent swipe to navigate
        let touchStartX = 0;
        let touchEndX = 0;
        
        document.addEventListener('touchstart', function(e) {
            touchStartX = e.changedTouches[0].screenX;
        }, false);
        
        document.addEventListener('touchend', function(e) {
            touchEndX = e.changedTouches[0].screenX;
            if (touchStartX - touchEndX > 50) {
                // Swiped left
                e.preventDefault();
            } else if (touchEndX - touchStartX > 50) {
                // Swiped right
                e.preventDefault();
            }
        }, false);

        // Prevent context menu (right-click menu)
        document.addEventListener('contextmenu', function(e) {
            e.preventDefault();
        }, false);

        // Prevent refresh
        document.addEventListener('keydown', function(e) {
            if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
                e.preventDefault();
            }
        }, false);

        // Prevent refresh on mobile pull-down
        let touchStartY = 0;
        document.addEventListener('touchstart', function(e) {
            touchStartY = e.touches[0].clientY;
        }, {passive: false});

        document.addEventListener('touchmove', function(e) {
            const touchY = e.touches[0].clientY;
            const touchYDelta = touchY - touchStartY;
            
            if (touchYDelta > 0 && window.scrollY === 0) {
                e.preventDefault();
            }
        }, {passive: false});

        const fetchWithTimeout = (url, options, timeout = 5000) => {
            console.log(`fetchWithTimeout called for ${url}`);
            return Promise.race([
                fetch(url, options),
                new Promise((_, reject) => 
                    setTimeout(() => {
                        console.log(`Timeout reached for ${url}`);
                        reject(new Error('Fetch timeout'));
                    }, timeout)
                )
            ]);
        };

        function fetchWithRetry(url, options, maxRetries = 3) {
            console.log(`fetchWithRetry called for ${url}, maxRetries: ${maxRetries}`);
            return new Promise((resolve, reject) => {
                const attempt = (retryCount) => {
                    console.log(`Attempt ${maxRetries - retryCount + 1} for ${url}`);
                    fetchWithTimeout(url, options)
                        .then(response => {
                            console.log(`Successful fetch for ${url}`);
                            resolve(response);
                        })
                        .catch((error) => {
                            console.error(`Error in fetch attempt for ${url}:`, error);
                            if (retryCount === 0) {
                                console.log(`All retries exhausted for ${url}`);
                                reject(error);
                            } else {
                                console.log(`Retrying fetch for ${url}. ${retryCount} attempts left.`);
                                attempt(retryCount - 1);
                            }
                        });
                };
                attempt(maxRetries);
            });
        }

        function loadResource(src) {
            console.log(`loadResource called for ${src}`);
            return new Promise((resolve, reject) => {
                if (src.endsWith('.json')) {
                    console.log(`Fetching JSON: ${src}`);
                    fetchWithRetry(src, {})
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }
                            console.log(`JSON fetch successful: ${src}`);
                            return response.json();
                        })
                        .then(data => {
                            console.log(`JSON parsed successfully: ${src}`);
                            resolve(data);
                        })
                        .catch(error => {
                            console.error(`Error loading JSON ${src}:`, error);
                            reject(error);
                        });
                } else {
                    console.log(`Loading script: ${src}`);
                    const script = document.createElement('script');
                    script.src = src;
                    script.onload = () => {
                        console.log(`Script loaded successfully: ${src}`);
                        resolve();
                    };
                    script.onerror = () => {
                        console.error(`Error loading script ${src}`);
                        reject(new Error(`Failed to load ${src}`));
                    };
                    document.head.appendChild(script);
                }
            });
        }

        function createPlot(elementId, data) {
            console.log(`createPlot called for ${elementId}`);
            data.layout.autosize = true;
            data.layout.width = window.innerWidth;
            data.layout.margin = {l: 40, r: 5, t: 10, b: 10};
            
            if (elementId === 'top-plotly-chart') {
                data.layout.height = window.innerHeight * 0.2;
                data.layout.dragmode = false;
                data.layout.hovermode = false;
                data.config = {
                    scrollZoom: false,
                    displayModeBar: false,
                    responsive: true
                };
            } else {
                data.layout.height = window.innerHeight * 0.8;
                data.layout.dragmode = 'zoom';
                data.config = {
                    scrollZoom: false,
                    displayModeBar: false,
                    responsive: true
                };
            }

            return Plotly.newPlot(elementId, data.data, data.layout, data.config)
                .then(() => console.log(`Plot created successfully for ${elementId}`))
                .catch(error => console.error(`Error creating plot for ${elementId}:`, error));
        }

        function resizePlots() {
            console.log('resizePlots called');
            const newWidth = window.innerWidth;
            Plotly.relayout('top-plotly-chart', {
                width: newWidth,
                height: window.innerHeight * 0.2
            }).catch(error => console.error('Error resizing top chart:', error));
            Plotly.relayout('main-plotly-chart', {
                width: newWidth,
                height: window.innerHeight * 0.8
            }).catch(error => console.error('Error resizing main chart:', error));
        }

        console.log('Starting to load resources');
        Promise.all([
            loadResource('http://localhost:8010/plotly-2.34.0.min.js'),
            loadResource('http://localhost:8010/TopPlotly.json'),
            loadResource('http://localhost:8010/plotly.json')
        ])
        .then(([_, topJsonData, mainJsonData]) => {
            console.log('All resources loaded successfully');
            return Promise.all([
                createPlot('top-plotly-chart', topJsonData),
                createPlot('main-plotly-chart', mainJsonData)
            ]);
        })
        .then(() => {
            console.log('Both plots created successfully');
            window.addEventListener('resize', resizePlots);
            console.log('Resize event listener added');
        })
        .catch(error => console.error('Error in main execution:', error));

        console.log('Script execution completed');
    </script>
</body>
</html>
"""
        
        self.webview1.set_content(content=self.img_html, root_url="http://localhost:8010/")
        #self.bg3.image = toga.Image(output_file)
        #self.show_page3()
        #self.bg3.refresh()

        print("Wrapper done")
        return
    
    async def get_textbox_values(self, widget):
        global wella
        global attrib
        global model
        global unitchoice
        global UCSs
        global forms
        try:
            try:
                with open(unitpath, 'r') as f:
                    reader = csv.reader(f)
                    unitchoice = next(reader)
                    unitchoice = [int(x) for x in unitchoice]  # Convert strings to integers
            except:
                unitchoice = [0,0,0,0,0] #Depth, pressure,gradient, strength, temperature
            
            try:
                UCSs = pd.read_csv(output_ucs)
            except:
                UCSs = None
            try:
                forms = pd.read_csv(output_forms)
            except:
                forms = None
                
            self.progress.text = "Status: Calculating, Standby"
            self.getwelldev()
            data = pd.read_csv(modelpath, index_col=False)
            data_into_list = data.values.tolist()
            print(data_into_list)
            model = data_into_list[0]
            tail = model[-4:]
            tv = [textbox.value for textbox in self.textboxes]
            #self.bg3.image = toga.Image('BG1.png')
            self.bg4.image = toga.Image('BG1.png')
            model = tv + tail
            print(model)
            with open(modelpath, 'w') as file:
                file.write(modelheader + '\n')
                for item in model:
                    file.write(str(item) + ",")
            print("Great Success!! :D")
            
            self.page3_btn1.enabled = False
            self.page3_btn2.enabled = False
            self.page3_btn3.enabled = False
            self.page3_btn4.enabled = False
            self.page3_btn5.enabled = False
            
            global mwvalues
            global flowgradvals
            global fracgradvals
            global flowpsivals
            global fracpsivals
            
            mwvalues = self.get_depth_mw_data_values()
            fracgradvals = self.get_frac_grad_data_values()[0]
            flowgradvals = self.get_flow_grad_data_values()[0]
            fracpsivals = self.get_frac_grad_data_values()[1]
            flowpsivals = self.get_flow_grad_data_values()[1]
            
            with open(algopath, 'r') as file:
                data = file.read().strip()  # Read the file and remove any trailing newline
                prog_opts = [int(float(num)) for num in data.split(',')]  # Split by commas and convert to integers

            
            print("model_fin: ",model)
            
            self.progress.start()
            self.stop_server()
            loop = asyncio.get_running_loop()            
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(
                    pool, 
                    self.start_plotPPzhang_thread, 
                    loop, wella,{
                        'rhoappg': float(model[0]),
                        'a': float(model[1]),
                        'nu': float(model[2]),
                        'tecb': float(model[3]),
                        'lamb': float(model[4]),
                        'dtml': float(model[5]), 
                        'dtmt': float(model[6]),
                        'ul_exp': float(model[7]),
                        'ul_depth': float(model[8]),
                        'res0': (float(model[9])),
                        'be': (float(model[10])),
                        'ne': (float(model[11])),
                        'dex0':(float(model[12])),
                        'de':(float(model[13])),
                        'nde':(float(model[14])),
                        'underbalancereject': float(model[15]),
                        'doi': (float(model[16])*ureg(ul[unitchoice[0]])).to('metre').magnitude, 
                        'offset': float(model[17]), 
                        'strike': float(model[18]), 
                        'dip': float(model[19]),
                        'sfs': float(model[20]),
                        'water': float(model[21]),
                        'mudtemp': (float(model[22])*ureg(ut[unitchoice[4]])).to('degC').magnitude if float(model[16]) !=0 else 0,
                        'window': int(float(model[23])),
                        'zulu': (float(model[24])*ureg(ul[unitchoice[0]])).to('metre').magnitude,
                        'tango': (float(model[25])*ureg(ul[unitchoice[0]])).to('metre').magnitude,
                        'lala': -1.0, 
                        'lalb': 1.0, 
                        'lalm': 5, 
                        'lale': 0.5, 
                        'lall': 5, 
                        'horsuda': 0.77, 
                        'horsude': 2.93,
                        'unitchoice': unitchoice,
                        'ureg': ureg,
                        'mwvalues': mwvalues,
                        'flowgradvals': flowgradvals,
                        'fracgradvals': fracgradvals,
                        'flowpsivals': flowpsivals,
                        'fracpsivals': fracpsivals,
                        'attrib': attrib,
                        'flags': flags,
                        'UCSs': UCSs,
                        'forms': forms,
                        "lithos": lithos,
                        "user_home": user_home,
                        "paths": path_dict,
                        "program_option": prog_opts
                    }          
                )
            print(model[3])
            print("Calculation complete")
        except Exception as e:
            error_message = str(e)
            traceback_str = traceback.format_exc()
            print(f"Error in get_textbox_values: {error_message}")
            self.main_window.error_dialog('Error:', error_message)
            await self.onplotfuckup(traceback_str)  # Ensure cleanup happens even if there's an error
    
    
    def wellisvertical(self,widget):        
        self.main_window.content = self.page2
        #self.bg3.refresh()
        
        
    def start_server(self):
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)

        self.server = socketserver.TCPServer(('localhost', 8000), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        print("server 1 started")
    
    def start_server2(self):
        # Store the local directory
        local_dir = os.getcwd()

        # Define the request handler class with dual directories
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                # Construct the full path for the local and output directories
                local_path = os.path.join(local_dir, self.path.lstrip('/'))
                output_path = os.path.join(output_dir, self.path.lstrip('/'))
                print(output_path)
                if os.path.exists(local_path) and os.path.isfile(local_path):
                    # Serve from the local directory if file exists
                    self.directory = local_dir
                elif os.path.exists(output_path) and os.path.isfile(output_path):
                    # Serve from the output directory if file exists
                    self.directory = output_dir
                else:
                    # Default to the local directory if file not found
                    self.directory = local_dir

                print(f"Received GET request for {self.path}, serving from {self.directory}")
                super().do_GET()

            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                super().end_headers()

        # Create the server
        self.server2 = ThreadingHTTPServer(('localhost', 8010), Handler)

        # Start the server in a separate thread
        self.server2_thread = threading.Thread(target=self.server2.serve_forever)
        #self.server2_thread.daemon = True  # This ensures the server thread exits when the main thread does
        self.server2_thread.start()
        print("server 2 started, serving files from both local and:", output_dir)
   
    def stop_server(self):
        """
        if hasattr(self, 'server'):
            #self.server.shutdown()
            print("shutdown")
            self.server.server_close()
            print("server close")
            #self.server_thread.join()
            print("thread join")
            del self.server_thread
            print("del server")
            del self.server
            print("server stopped and thread joined")
        """
        if hasattr(self, 'server2'):
            #self.server.shutdown()
            print("shutdown")
            self.server2.server_close()
            print("server2 close")
            #self.server_thread.join()
            print("thread join")
            del self.server2_thread
            print("del server2")
            del self.server2
            print("server2 stopped and thread joined")

def readDevFromAsciiHeader(devpath, delim = r'[ ,	]'):
    dev=pd.read_csv(devpath, sep=delim)
    dheader = list(dev.columns)
    return dheader
def readLithoFromAscii(lithopath, delim = r'[ ,	]'):
    global lithos
    litho=pd.read_csv(lithopath, sep=delim)
    lithos=litho
    lithoheader = list(litho.columns)
    return litho

def readUCSFromAscii(ucspath, delim = r'[ ,	]'):
    global UCSs
    ucs=pd.read_csv(ucspath, sep=delim)
    UCSs=ucs
    ucsheader = list(ucs.columns)
    return ucs

def readFlagFromAscii(flagpath, delim = r'[ ,	]'):
    global flags
    flag=pd.read_csv(flagpath, sep=delim)
    flags=flag
    flagheader = list(flag.columns)
    return flag

def readFormFromAscii(formpath, delim = r'[ ,	]'):
    global forms
    form=pd.read_csv(formpath, sep=delim)
    forms=form
    formheader = list(form.columns)
    return form




def main():
    app = MyApp('Stresslog', 'in.rocklab.stresslog')
    return app

if __name__ == "__main__":
    app = MyApp("Stresslog", "in.rocklab.stresslog")
    app.stop_server()
    app.start_server2()
    app.main_loop()