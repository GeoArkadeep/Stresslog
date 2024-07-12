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

import matplotlib
matplotlib.use("svg")

from matplotlib import pyplot as plt    
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
    
from manage_preferences import show_preferences_window
from editor import custom_edit

user_home = os.path.expanduser("~/Documents")
app_data = os.getenv("APPDATA")
output_dir = os.path.join(user_home, "ppp_app_plots")
output_dir1 = os.path.join(user_home, "ppp_app_data")
input_dir = os.path.join(user_home, "ppp_app_models")
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

output_file = os.path.join(output_dir, "PlotFigure.png")
output_fileS = os.path.join(output_dir, "PlotStability.png")
output_fileSP = os.path.join(output_dir, "PlotPolygon.png")
output_fileVec = os.path.join(output_dir, "PlotVec.png")
output_fileBHI = os.path.join(output_dir, "PlotBHI.png")
output_fileHoop = os.path.join(output_dir, "PlotHoop.png")
output_fileFrac = os.path.join(output_dir, "PlotFrac.png")
output_fileAll = os.path.join(output_dir, "PlotAll.png")
output_file2 = os.path.join(output_dir1, "output.csv")
output_file3 = os.path.join(output_dir1, "output.las")
modelpath = os.path.join(input_dir, "model.csv")
aliaspath = os.path.join(input_dir, "alias.txt")
unitpath = os.path.join(input_dir, "units.txt")
stylespath = os.path.join(input_dir, "styles.txt")
pstylespath = os.path.join(input_dir, "pstyles.txt")
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

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
ureg.define('ppg = 0.051948 psi/foot')
ureg.define('sg = 0.4335 psi/foot = gcc')
ureg.define('ksc = 1.0000005979/0.0703069999987293 psi = KSC = KSc = KsC = ksC = Ksc')

try:
    with open(unitpath, 'r') as f:
        reader = csv.reader(f)
        unitchoice = next(reader)
        unitchoice = [int(x) for x in unitchoice]  # Convert strings to integers
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

modelheader = "RhoA,AMC_exp,EATON_fac,tec_fac,NCT_exp,dtML,dtMAT,UL_exp,UL_depth,re_sub,A_dep,SHM_azi,Beta,Gamma,perm_cutoff,w_den,MudTempC,window,start,stop,nu_shale,nu_sst,nu_lst,dt_lst"
defaultmodel = "17,0.8,0.35,0,0.0008,250,60,0.0008,0,1,3500,0,0,0,0.35,1.025,60,21,0,2900,0.32,0.27,0.25,65"

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
        # This method will be called when the preferences command is activated
        show_preferences_window(aliaspath, stylespath, pstylespath,unitdict,unitpath)
    def custom_edit_ucs(self, widget):
        asyncio.create_task(self.run_custom_ucs())

    async def run_custom_ucs(self):
        global UCSs
        UCSs = await custom_edit(
            self, 
            UCSs, 
            ["MD", "UCS"], ["Depth", "Strength"], 
            unitdict,["Metres","MPa"],[float,float],ureg
        )
        print(UCSs)
        
    def custom_edit_forms(self, widget):
        asyncio.create_task(self.run_custom_forms())

    async def run_custom_forms(self):
        global forms
        formunitdict = {"Depth": ["Metres", "Feet"],
            "None": [""]}
        forms = await custom_edit(
            self, 
            forms, 
            ["Top TVD", "Number", "Formation Name", "GR Cut", "Struc.Top", "Struc.Bottom", "CentroidRatio", "OWC", "GOC", "Coeff.Vol.Therm.Exp.","SHMax Azim.", "SVDip", "SVDipAzim","Tectonic Factor","InterpretedSH/Sh","Biot"], ["Depth","None","None","None", "Depth","Depth","None","Depth","Depth","None"], 
            formunitdict,["Metres","","","","Metres","Metres","","Metres","Metres","","","","","","",""],[float,int,str,float,float,float,float,float,float,float,float,float,float,float,float,float,float],ureg
        )
        print(forms)
        
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

        # Explicitly set it as the preferences command
        #self.set_preferences(PREFERENCES)
        
        custom_edit_ucs = toga.Command(
            self.custom_edit_ucs,
            text='Edit UCS data',
            shortcut=toga.Key.MOD_1 + 'u',
            group=toga.Group.EDIT
        )
        self.commands.add(custom_edit_ucs)
        
        custom_edit_forms = toga.Command(
            self.custom_edit_forms,
            text='Edit Formation data',
            shortcut=toga.Key.MOD_1 + 'f',
            group=toga.Group.EDIT
        )
        self.commands.add(custom_edit_forms)

        
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
            
            iv_label = toga.Label("Casing volume (bbl/100ft)", style=Pack(padding_right=2,text_direction='rtl'))
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

        self.bg3 = BackgroundImageView("BG2.png", style=Pack(flex=1))
        plot_and_data_box.add(self.bg3)

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
            
            {'label': 'NCT Exp', 'default_value': str(model[4])},
            {'label': 'DTml (us/ft)', 'default_value': str(model[5])},
            {'label': 'DTmat (us/ft)', 'default_value': str(model[6])},
            {'label': 'Unloading Exp', 'default_value': str(model[7])},
            {'label': 'Unloading Depth', 'default_value': "0"},
            {'label': 'PP Gr. L.Limit', 'default_value': str(model[9])},

            {'label': 'Analysis TVD', 'default_value': "0"},
            {'label': 'Fast Shear Azimuth', 'default_value': str(model[11])},
            {'label': 'Dip Azim.', 'default_value': str(model[12])},
            {'label': 'Dip Angle', 'default_value': str(model[13])},
            
            {'label': 'ShaleFlag Cutoff', 'default_value': str(model[14])},
            {'label': 'WaterDensity', 'default_value': str(model[15])},
            {'label': 'MudTemp', 'default_value': str(model[16])},
            
            {'label': 'Window', 'default_value': str(model[17])},
            {'label': 'Start', 'default_value': str(model[18])},
            {'label': 'Stop', 'default_value': str(model[19])}
            
            
            
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
        for i in range(6):
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
        right_pane_container = toga.ScrollContainer(style=Pack(direction='row',flex=1))

        # Create a box for the progress bar and image
        #right_pane_box = toga.Box(style=Pack(direction=ROW, flex=1))
        
        # Adjust the ScrollContainer to handle overflow properly
        #right_pane_container.content = right_pane_box
        right_pane_container.horizontal = False
        girth = 1000

        # Add the image display to the right pane box
        my_image = toga.Image("BG2.png")
        self.bg3 = toga.ImageView(my_image, style=Pack(direction='row',flex=1))
        #right_pane_container.add(self.bg3)
        right_pane_container.content = self.bg3

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
        
        self.main_window = toga.MainWindow(title=self.formal_name,size=[1440,720])
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
                iv = float(iv_entry.value)
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
            spacing = ((wella.df().index.values[9]-wella.df().index.values[0])/9)
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
            spacing = ((wella.df().index.values[9]-wella.df().index.values[0])/9)
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
        wella.unify_basis(keys=None, alias=None, basis=md)
        
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
            laspath_dialog = await self.main_window.open_file_dialog(title="Select a las file",file_types=['las'], multiselect=False)
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
            
        else:
            print("No litho file loaded")

    def on_result3(self, widget):
        global h3, ucspath
        if ucspath is not None:               
            h3 = readUCSFromAscii(ucspath)
            print("Loaded ucs file:", ucspath)
            print(h3)           
            
        else:
            print("No ucs file loaded")
    
    def on_result4(self, widget):
        global h4, flagpath
        if flagpath is not None:               
            h4 = readFlagFromAscii(flagpath)
            print("Loaded flag file:", flagpath)
            print(h4)           
            
        else:
            print("No flag file loaded")

    def on_result5(self, widget):
        global h5, formpath
        if formpath is not None:               
            h5 = readFormFromAscii(formpath)
            print("Loaded formation file:", formpath)
            print(h5)           
            
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
            kwargs['mudtemp']
        )
    
    def plotppwrapper(self, loop, *args, **kwargs):
        print("thread spawn")
        try:
            result = self.plotPPzhang_wrapper(*args, **kwargs)
        except Exception as e:
            self.main_window.error_dialog('Error:', str(e))
            er = traceback.format_exc
            print(f"Error in thread: {e}")
        asyncio.run_coroutine_threadsafe(self.onplotfinish(), loop)
        print("Thread despawn")
        return
    
    def start_plotPPzhang_thread(self, loop, *args, **kwargs):
        thread = threading.Thread(target=self.plotppwrapper, args=(loop, *args), kwargs=kwargs)
        thread.start()
        return
    
    async def onplotfinish(self):
        self.bg3.image = toga.Image(output_file)
        #self.bg3.refresh()
        self.page3_btn1.enabled = True
        self.page3_btn2.enabled = True
        self.page3_btn3.enabled = True
        self.page3_btn4.enabled = True
        self.progress.stop()
        if float(model[10]) > 0:
            self.page3_btn5.enabled = True
            self.bg4.image = toga.Image(output_fileAll)
        else:
            self.page3_btn5.enabled = False
        print("Wrapper done")
        return
    
    async def get_textbox_values(self, widget):
        global wella
        global attrib
        global model
        global unitchoice
        try:
            with open(unitpath, 'r') as f:
                reader = csv.reader(f)
                unitchoice = next(reader)
                unitchoice = [int(x) for x in unitchoice]  # Convert strings to integers
        except:
            unitchoice = [0,0,0,0,0] #Depth, pressure,gradient, strength, temperature

        
        self.progress.text = "Status: Calculating, Standby"
        self.getwelldev()
        data = pd.read_csv(modelpath, index_col=False)
        data_into_list = data.values.tolist()
        print(data_into_list)
        model = data_into_list[0]
        tail = model[20:23]
        tv = [textbox.value for textbox in self.textboxes]
        self.bg3.image = toga.Image('BG1.png')
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
        
        self.progress.start()

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
                    'underbalancereject': float(model[9]),
                    'doi': (float(model[10])*ureg(ul[unitchoice[0]])).to('metre').magnitude, 
                    'offset': float(model[11]), 
                    'strike': float(model[12]), 
                    'dip': float(model[13]),
                    'sfs': float(model[14]),
                    'water': float(model[15]),
                    'mudtemp': (float(model[16])*ureg(ut[unitchoice[4]])).to('degC').magnitude if float(model[16]) !=0 else 0,
                    'window': int(float(model[17])),
                    'zulu': (float(model[18])*ureg(ul[unitchoice[0]])).to('metre').magnitude,
                    'tango': (float(model[19])*ureg(ul[unitchoice[0]])).to('metre').magnitude
                }          
            )
        print(model[3])
        print("Calculation complete")        
    
    
    def wellisvertical(self,widget):
        global depth_track, finaldepth
        # Print the updated self.output list to the console
        print("Blah")
        depth_track = wella.df().index.values
        print(depth_track)
        print(len(depth_track))
        
        start_depth = wella.df().index[0]
        final_depth = wella.df().index[-1]
        spacing = ((wella.df().index.values[9]-wella.df().index.values[0])/9)
        print("Sample interval is :",spacing)
        padlength = int(start_depth/spacing)
        print(padlength)
        padval = np.zeros(padlength)
        i = 1
        while(i<padlength):
            padval[-i] = start_depth-(spacing*i)
            i+=1
        print("pad depths: ",padval)
        print("pad length is :",padlength,", Total length is :",padlength+len(depth_track))
        
        md = depth_track
        md2 =  np.append(padval,md)
        print("new track length is :",len(md2))
        #md[0] = 0
        #md[0:padlength-1] = padval[0:padlength-1]
        inc = np.zeros(len(depth_track)+padlength)
        azm = np.zeros(len(depth_track)+padlength)
        dz = [md2,inc,azm]
        dz = np.transpose(dz)
        dz = pd.DataFrame(dz)
        #dz = dz.dropna()
        print(dz)
        finaldepth = dz.to_numpy()[-1][0]
        print("Final depth is ",finaldepth)
        wella.location.add_deviation(dz, wella.location.td)
        tvdg = wella.location.tvd
        md3 = wella.location.md
        from welly import curve
        MD = curve.Curve(md3, mnemonic='MD',units='m', index = md3)
        wella.data['MD'] =  MD
        TVDM = curve.Curve(tvdg, mnemonic='TVDM',units='m', index = md3)
        wella.data['TVDM'] =  TVDM
        
        wella.unify_basis(keys=None, alias=None, basis=md3)
        self.bg3.image = toga.Image('BG1.png')
        #plotPPzhang(wella,self)

        print("Great Success!! :D")
        image_path = 'PlotFigure.png'
        #self.bg3.image = toga.Image(output_file)
        #Clock.schedule_once(lambda dt: self.refresh_plot(image_path), 5)
        #print(self.output)
        self.main_window.content = self.page2
        #self.bg3.refresh()

"""
def getComp(well):
    alias = read_aliases_from_file()
    header = well._get_curve_mnemonics()
    #print(header)
    alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
    alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
    alias['ssonic'] = [elem for elem in header if elem in set(alias['ssonic'])]
    alias['resdeep'] = [elem for elem in header if elem in set(alias['resdeep'])]
    alias['resshal'] = [elem for elem in header if elem in set(alias['resshal'])]
    alias['density'] = [elem for elem in header if elem in set(alias['density'])]
    alias['neutron'] = [elem for elem in header if elem in set(alias['neutron'])]
    #alias['pe'] = [elem for elem in header if elem in set(alias['pe'])]
"""    
    
    


def getNu(well, nun):
    import math
    alias = read_aliases_from_file()
    header = well._get_curve_mnemonics()
    #print(header)
    alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
    alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
    alias['ssonic'] = [elem for elem in header if elem in set(alias['ssonic'])]
    alias['resdeep'] = [elem for elem in header if elem in set(alias['resdeep'])]
    alias['resshal'] = [elem for elem in header if elem in set(alias['resshal'])]
    alias['density'] = [elem for elem in header if elem in set(alias['density'])]
    alias['neutron'] = [elem for elem in header if elem in set(alias['neutron'])]
    
    vp = 1/well.data[alias['sonic'][0]].values
    vs = 1/well.data[alias['ssonic'][0]].values
    vpvs = vp/vs
    nu = ((vpvs**2)-2)/((2*(vpvs**2))-2)
    nu = [x if not math.isnan(x) else nun for x in nu]
    nu = [x if not math.isnan(x) else nun for x in nu]
    nu = [x if not (x == float('inf') or x == float('-inf')) else nun for x in nu]
    print("nu: ", nu)
    
    return nu



def read_aliases_from_file(file_path=aliaspath):
    import json
    try:
        with open(file_path, 'r') as file:
            aliases = json.load(file)  # Use json.load() to parse the file content
        return aliases
    except:
        aliases = {
            'sonic': ['none', 'DTC', 'DT24', 'DTCO', 'DT', 'AC', 'AAC', 'DTHM'],
            'ssonic': ['none', 'DTSM'],
            'gr': ['none', 'GR', 'GRD', 'CGR', 'GRR', 'GRCFM'],
            'resdeep': ['none', 'HDRS', 'LLD', 'M2RX', 'MLR4C', 'RD', 'RT90', 'RLA1', 'RDEP', 'RLLD', 'RILD', 'ILD', 'RT_HRLT', 'RACELM'],
            'resshal': ['none', 'LLS', 'HMRS', 'M2R1', 'RS', 'RFOC', 'ILM', 'RSFL', 'RMED', 'RACEHM'],
            'density': ['none', 'ZDEN', 'RHOB', 'RHOZ', 'RHO', 'DEN', 'RHO8', 'BDCFM'],
            'neutron': ['none', 'CNCF', 'NPHI', 'NEU'],
            'pe': ['none','PE']
        }
        # Convert aliases dictionary to JSON string and write to the file
        with open(file_path, 'w') as file:
            json.dump(aliases, file, indent=4)
        return aliases


def read_styles_from_file(minpressure, maxchartpressure, pressure_units, strength_units, gradient_units, file_path=stylespath):
    
    def convert_value(value, from_unit, to_unit):
        return (value * ureg(from_unit)).to(to_unit).magnitude

    try:
        with open(file_path, 'r') as file:
            styles = json.load(file)
    except:
        styles = {
            'dalm': {"color": "green", "linewidth": 1.5, "style": '-', "track": 1, "left": 300, "right": 50, "type": 'linear', "unit": "us/ft"},
            'dtNormal': {"color": "blue", "linewidth": 1.5, "style": ':', "track": 1, "left": 300, "right": 50, "type": 'linear', "unit": "us/ft"},
            'mudweight': {"color": "brown", "linewidth": 1.5, "style": '-', "track": 2, "left": 0, "right": 3, "type": 'linear', "unit": "gcc"},
            'fg': {"color": "blue", "linewidth": 1.5, "style": '-', "track": 2, "left": 0, "right": 3, "type": 'linear', "unit": "gcc"},
            'pp': {"color": "red", "linewidth": 1.5, "style": '-', "track": 2, "left": 0, "right": 3, "type": 'linear', "unit": "gcc"},
            'sfg': {"color": "olive", "linewidth": 1.5, "style": '-', "track": 2, "left": 0, "right": 3, "type": 'linear', "unit": "gcc"},
            'obgcc': {"color": "lime", "linewidth": 1.5, "style": '-', "track": 2, "left": 0, "right": 3, "type": 'linear', "unit": "gcc"},
            'fgpsi': {"color": "blue", "linewidth": 1.5, "style": '-', "track": 3, "left": minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'ssgHMpsi': {"color": "pink", "linewidth": 1.5, "style": '-', "track": 3, "left": minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'obgpsi': {"color": "green", "linewidth": 1.5, "style": '-', "track": 3, "left": minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'hydropsi': {"color": "aqua", "linewidth": 1.5, "style": '-', "track": 3, "left": minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'pppsi': {"color": "red", "linewidth": 1.5, "style": '-', "track": 3, "left": minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'mudpsi': {"color": "brown", "linewidth": 1.5, "style": '-', "track": 3, "left": minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'sgHMpsiL': {"color": "lime", "linewidth": 0.25, "style": ':', "track": 3, "left":minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'sgHMpsiU': {"color": "orange", "linewidth": 0.25, "style": ':', "track": 3, "left": minpressure, "right": maxchartpressure, "type": 'linear', "unit": "psi"},
            'slal': {"color": "blue", "linewidth": 1.5, "style": '-', "track": 4, "left": 0, "right": 100, "type": 'linear', "unit": "MPa"},
            'shorsud': {"color": "red", "linewidth": 1.5, "style": '-', "track": 4, "left": 0, "right": 100, "type": 'linear', "unit": "MPa"},
            'GR': {"color": "green", "linewidth": 0.25, "style": '-', "track": 0, "left": 0, "right": 150, "type": 'linear', "unit": "gAPI", "fill":'none', "fill_between": {"reference": "GR_CUTOFF", "colors": ["green", "yellow"], "colorlog":"obgcc","cutoffs":[1.8,2.67,2.75],"cmap":'Set1_r'}},
            'GR_CUTOFF': {"color": "black", "linewidth": 0, "style": '-', "track": 0, "left": 0, "right": 150, "type": 'linear', "unit": "gAPI"}
        }

    # Update tracks based on input units
    for key, value in styles.items():
        if value['track'] == 2:  # Gradient track
            if value['unit'] != gradient_units:
                value['left'] = round(convert_value(value['left'], value['unit'], gradient_units),1)
                value['right'] = round(convert_value(value['right'], value['unit'], gradient_units),1)
                value['unit'] = gradient_units
        elif value['track'] == 3:  # Pressure track
            if value['unit'] != pressure_units:
                value['unit'] = pressure_units
            value['left'] = round(convert_value(minpressure, 'psi', pressure_units))
            value['right'] = round(convert_value(maxchartpressure, 'psi', pressure_units))
        elif value['track'] == 4:  # Strength track
            if value['unit'] != strength_units:
                value['left'] = round(convert_value(value['left'], value['unit'], strength_units))
                value['right'] = round(convert_value(value['right'], value['unit'], strength_units))
                value['unit'] = strength_units

    # Write updated styles back to file
    with open(file_path, 'w') as file:
        json.dump(styles, file, indent=4)

    return styles

def read_pstyles_from_file(minpressure, maxchartpressure, pressure_units, strength_units, gradient_units, file_path=pstylespath):
    
    def convert_value(value, from_unit, to_unit):
        return (value * ureg(from_unit)).to(to_unit).magnitude

    try:
        with open(file_path, 'r') as file:
            pstyles = json.load(file)
    except:
        pstyles = {
                    'frac_grad': {'color': 'dodgerblue', 'pointsize': 100, 'symbol': 4, 'track': 2, 'left': 0, 'right': 3, 'type': 'linear', 'unit': 'gcc'},
                    'flow_grad': {'color': 'orange', 'pointsize': 100, 'symbol': 5, 'track': 2, 'left': 0, 'right': 3, 'type': 'linear', 'unit': 'gcc'},
                    'frac_psi': {'color': 'dodgerblue', 'pointsize': 100, 'symbol': 4, 'track': 3, 'left': minpressure, 'right': maxchartpressure, 'type': 'linear', 'unit': 'psi'},
                    'flow_psi': {'color': 'orange', 'pointsize': 100, 'symbol': 5, 'track': 3, 'left': minpressure, 'right': maxchartpressure, 'type': 'linear', 'unit': 'psi'},
                    'ucs': {'color': 'lime', 'pointsize': 30, 'symbol': 'o', 'track': 4, 'left': 0, 'right': 100, 'type': 'linear', 'unit': 'MPa'}
        }

    # Update tracks based on input units
    for key, value in pstyles.items():
        if value['track'] == 2:  # Gradient track
            if value['unit'] != gradient_units:
                value['left'] = round(convert_value(value['left'], value['unit'], gradient_units),1)
                value['right'] = round(convert_value(value['right'], value['unit'], gradient_units),1)
                value['unit'] = gradient_units
        elif value['track'] == 3:  # Pressure track
            if value['unit'] != pressure_units:
                value['unit'] = pressure_units
            value['left'] = round(convert_value(minpressure, 'psi', pressure_units))
            value['right'] = round(convert_value(maxchartpressure, 'psi', pressure_units))
        elif value['track'] == 4:  # Strength track
            if value['unit'] != strength_units:
                value['left'] = round(convert_value(value['left'], value['unit'], strength_units))
                value['right'] = round(convert_value(value['right'], value['unit'], strength_units))
                value['unit'] = strength_units

    # Write updated styles back to file
    with open(file_path, 'w') as file:
        json.dump(pstyles, file, indent=4)

    return pstyles


def pad_val(array_like,value):
    array = array_like.copy()

    nans = np.isnan(array)

    def get_x(a):
        return a.nonzero()[0]

    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

    return array

def find_nearest_depth(array,value):
    import math
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return [idx-1,array[idx-1]]
    else:
        return [idx,array[idx]]

def interpolate_nan(array_like):
    array = array_like.copy()

    nans = np.isnan(array)

    def get_x(a):
        return a.nonzero()[0]

    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

    return array


def plotPPzhang(well,rhoappg = 16.33, lamb=0.0008, ul_exp = 0.0008, ul_depth = 0, a = 0.630, nu = 0.4, sfs = 1.0, window = 1, zulu=0, tango=2000, dtml = 210, dtmt = 60, water = 1.0, underbalancereject = model[13], tecb = 0, doi = 0, offset = 0, strike = 0, dip = 0, mudtemp = 0, lala = -1.0, lalb = 1.0, lalm = 5, lale = 0.5, lall = 5, horsuda = 0.77, horsude = 2.93):
    global unitchoice
    alias = read_aliases_from_file()
    from welly import Curve
    #print(alias)
    print(well.uwi,well.name)
    #print(well.location.location)
    start_depth = wella.df().index[0]
    final_depth = wella.df().index[-1]
    global finaldepth
    plt.clf()
    #well.location.plot_3d()
    #well.location.plot_plan()
    
    from BoreStab import getEuler
    
    if strike != 0 or dip != 0:
        tilt, tiltgamma = getEuler(offset,strike,dip)
        print("Alpha :",offset,", Beta: ",tilt,", Gamma :",tiltgamma)
    else:
    
        tilt = 0
        tiltgamma = 0
    
    header = well._get_curve_mnemonics()
    print(header)
    alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
    alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
    alias['ssonic'] = [elem for elem in header if elem in set(alias['ssonic'])]
    alias['resdeep'] = [elem for elem in header if elem in set(alias['resdeep'])]
    alias['resshal'] = [elem for elem in header if elem in set(alias['resshal'])]
    alias['density'] = [elem for elem in header if elem in set(alias['density'])]
    alias['neutron'] = [elem for elem in header if elem in set(alias['neutron'])]
    #alias['pe'] = [elem for elem in header if elem in set(alias['pe'])]
    
    global mwvalues
    global flowgradvals
    global fracgradvals
    global flowpsivals
    global fracpsivals
    global attrib
    
    detail = mwvalues
    print(detail)
    i = 0
    mud_weight = []
    bht_point = []
    casing_dia = []
    while i<len(detail):
        mud_weight.append([detail[i][0],detail[i][1]])
        bht_point.append([detail[i][-1],detail[i][1]])
        casing_dia.append([detail[i][3],detail[i][1]])
        i+=1    
    print(mud_weight)
    first = [mud_weight[0][0],0]
    last = [mud_weight[-1][0],final_depth]
    top_bht =[15,0]
    bottom_bht = [float(attrib[5]),final_depth]
    frac_grad_data = fracgradvals
    flow_grad_data = flowgradvals
    frac_psi_data = fracpsivals
    flow_psi_data = flowpsivals
    
    mud_weight.insert(0,first)
    mud_weight.append(last)
    bht_point.insert(0,top_bht)
    bht_point.append(bottom_bht)
    print("MudWeights: ",mud_weight)
    print("BHTs: ",bht_point)
    print(len(bht_point))
    print (alias['sonic'])
    if alias['sonic'][0] == 'none':
        print("Without sonic log, no prediction possible")
        return
    vp = 0
    vs = 0
    vpvs = 0
    nu2 = []
    
    dt = well.data[alias['sonic'][0]]
    
    md = well.data['MD'].values
    try:
        nu2 = getNu(well, nu)
    except:
        nu2 = [nu] * (len(md))
    
    
    #kth = well.data['KTH']
        

    
    try:
        zden2 = well.data[alias['density'][0]].values
    except:
        zden2 = np.full(len(md),np.nan)
    try:
        gr = well.data[alias['gr'][0]].values
    except:
        gr = np.full(len(md),np.nan)
    try:
        cald = well.data[alias['cald'][0]].values
    except:
        cald = np.full(len(md),np.nan)
    try:
        cal2 = well.data[alias['cal2'][0]].values
    except:
        cal2 = np.full(len(md),np.nan)

    
    lradiff = np.full(len(md),np.nan)
    
    if alias['resshal'] != [] and alias['resdeep'] != []:
        rS = well.data[alias['resshal'][0]].values
        rD = well.data[alias['resdeep'][0]].values
        print(rD)
        #sd = gr.plot_2d(cmap='viridis_r', curve=True, lw=0.3, edgecolor='k')
        #sd2 = kth.plot_2d(cmap='viridis_r', curve=True, lw=0.3, edgecolor='k')
        #plt.xlim(0,150)
        rdiff = rD[:]-rS[:]
        rdiff[np.isnan(rdiff)] = 0
        #rdiff = interpolate_nan(rdiff,0)
        radiff = (rdiff[:]*rdiff[:])**0.5
        #plt.plot(radiff)
        #plt.yscale('log')
        #i = 0
        lradiff = radiff
        #while i<len(radiff):
        #    lradiff[i] = radiff[i]
        #    i+=1
        #print("Rdiff :",lradiff)
    
        rediff = Curve(lradiff, mnemonic='ResD',units='ohm/m', index=md, null=0)
        well.data['ResDif'] =  rediff
        print("sfs = :",sfs)
        shaleflag = rediff.block(cutoffs=sfs,values=(0,1)).values
        zoneflag = rediff.block(cutoffs=sfs,values=(0,1)).values
        print(shaleflag)
        #plt.plot(shaleflag)        
        ##plt.show()
    else:
        shaleflag = np.zeros(len(md))
        zoneflag = np.zeros(len(md))
    shaleflagN = (np.max(shaleflag)-shaleflag[:])/np.max(shaleflag)
    flag = Curve(shaleflag, mnemonic='ShaleFlag',units='ohm/m', index=md, null=0)
    zone = Curve(zoneflag, mnemonic='ShaleFlag',units='ohm/m', index=md, null=0)
    well.data['Flag'] =  flag
    
    #print(shaleflag)
    
    array = well.data_as_matrix(return_meta = False)
    dfa = well.df()
    dfa = dfa.dropna()
    print(dfa)
    dt = well.data[alias['sonic'][0]]
    zden = dfa[alias['density'][0]]
    header = well._get_curve_mnemonics()
    #header += 'MD'
    csvdf = pd.DataFrame(array, columns=header)
    
    tvd = well.data['TVDM'].values
    
    
    #tvdf[0:6] = [0,50,100,200,400,800]
    #tvd[0:padlength-1] = padval[0:padlength-1]
    tvdf = tvd*3.28084
    tvdm = tvdf/3.28084
    tvdm[-1] = tvd[-1]
    print("length tvdf:",len(tvdf),"length tvd:",len(tvd))
    print("tvdf:",tvdf)
    
    #ppg at mudline 
    #a = 0.630 #amoco exponent
    #nu = 0.4
    
    global lithos
    global UCSs
    global flags
    global forms
    
    glwd = float(attrib[1])
    glf = glwd*(3.28084)
    wdf = glwd*(-3.28084)
    if wdf<0:
        wdf=0
    print(attrib[1])
    well.location.gl = float(attrib[1])
    well.location.kb = float(attrib[0])
    mudweight = np.zeros(len(tvd))
    bt = np.zeros(len(tvd))
    delTempC = np.zeros(len(tvd))
    tempC = np.zeros(len(tvd))
    tempC[:] = np.nan
    tempG = np.zeros(len(tvd))
    #mudweight[:] = np.nan
    try:
        agf = (float(well.location.kb)-float(well.location.gl))*3.28084
    except AttributeError:
        agf = (float(attrib[0])-float(attrib[1]))*3.28084
    if(glwd>=0):
        if np.abs(well.location.kb)<np.abs(well.location.gl):
            agf = well.location.kb*3.28084
            well.location.kb = well.location.gl+well.location.kb
    if(glwd<0):
        agf = well.location.kb*3.28084
    
    print("Rig floor is ",well.location.kb*3.28084,"feet above MSL")
    #well.location.kb = well.location.gl+(agf/3.28084)
    tvdmsloffset = well.location.kb
    groundoffset = well.location.gl
    tvdmsl = tvd[:]-tvdmsloffset
    tvdbgl = 0
    tvdbgl = tvdmsl[:]+groundoffset
    tvdbglf = np.zeros(len(tvdbgl))
    tvdmslf = np.zeros(len(tvdmsl))
    wdfi = np.zeros(len(tvdmsl))
    lithotype = np.zeros(len(tvdbgl))
    nulitho = np.zeros(len(tvdbgl))
    dtlitho = np.zeros(len(tvdbgl))
    ilog = np.zeros(len(tvdbgl))
    formtvd = np.full(len(tvd),np.nan)
    hydrotvd = np.full(len(tvd),np.nan)
    hydroheight = np.full(len(tvd),np.nan)
    structop= np.full(len(tvd),np.nan)
    strucbot= np.full(len(tvd),np.nan)
    Owc= np.full(len(tvd),np.nan)
    Goc= np.full(len(tvd),np.nan)
    ttvd = np.full(len(tvd),np.nan)
    btvd = np.full(len(tvd),np.nan)
    btvd2 = np.full(len(tvd),np.nan)
    grcut = np.full(len(tvd),np.nan)
    alphas = np.full(len(tvd),offset)
    betas = np.full(len(tvd),strike)
    gammas = np.full(len(tvd),dip)
    tecB = np.full(len(tvd),tecb)
    SHsh = np.full(len(tvd),np.nan)
    biot = np.full(len(tvd),1)
    #cdtvd1 = np.full(len(tvd),np.nan)
    #cdtvd2 = np.full(len(tvd),np.nan)
    
    if float(attrib[5])==0:
        bht_point[-1][0] = (tvdbgl[-1]/1000)*30
    i = 0
    while i<len(bht_point):
        if bht_point[i][0] == 0:
            bht_point.pop(i)
        i+=1
    tgrads=np.array(bht_point)
    i = 0
    while i<len(tgrads):
        tgrads[i] = [(bht_point[i][0]-15)/(tvdbgl[find_nearest_depth(md,tgrads[i][1])[0]]/1000),tgrads[i][1]]
        i+=1
    tgrads[0][0] = tgrads[1][0]
    print("BHTs: ",bht_point)
    print("TGs: ",tgrads)
    
    if lithos is not None:
        lithot = lithos.values.tolist()
        firstlith = [0,0,0,0,0]
        lastlith = [final_depth,0,0,0,0]
        lithot.insert(0,firstlith)
        lithot.append(lastlith)
        
    if flags is not None:
        imagelog = flags.values.tolist()
        firstimg = [0,0]
        lastimg = [final_depth,0]
        imagelog.insert(0,firstimg)
        imagelog.append(lastimg)
    
    if forms is not None:
        formlist = forms.values.tolist()
        print("Formation Data: ", formlist)
        ttvdlist = np.transpose(formlist)[4]
        ftvdlist = np.transpose(formlist)[0]
        logbotlist = ftvdlist
        ttvdlist = np.append(0,ttvdlist)
        logbotlist = np.append(logbotlist,tvd[-1])
        fttvdlist = np.append(0,ftvdlist)
        difftvd = np.zeros(len(fttvdlist))
        hydrldiff = np.zeros(len(fttvdlist))
        htvdlist = np.zeros(len(fttvdlist))
        owclist = np.transpose(formlist)[7]
        owclist = np.append(0,owclist)
        goclist = np.transpose(formlist)[8]
        goclist = np.append(0,goclist)
        btlist = np.transpose(formlist)[9]
        btlist = np.append(btlist,btlist[-1])
        strucbotlist = np.transpose(formlist)[5]
        strucbotlist = np.append(ttvdlist[1],strucbotlist)
        logtoplist = np.append(0,ftvdlist)#,tvd[-1])
        ftvdlist = np.append(ftvdlist,tvd[-1])
        centroid_ratio_list = np.transpose(formlist)[6]
        centroid_ratio_list = np.append([0.5],centroid_ratio_list)
        grlist = np.transpose(formlist)[3]
        grlist = np.append(grlist[0],grlist)
        print(ftvdlist,btlist)
        centroid_ratio_list = centroid_ratio_list.astype(float)
        grlist = grlist.astype(float)
        alphalist = np.transpose(formlist)[10]
        alphalist = np.append(offset,alphalist)
        betalist = np.transpose(formlist)[11]
        betalist = np.append(strike,betalist)
        gammalist = np.transpose(formlist)[12]
        gammalist = np.append(dip,gammalist)
        
        tecBlist = np.transpose(formlist)[13]
        tecBlist = np.append(tecb,tecBlist)
        SHshlist = np.transpose(formlist)[14]
        SHshlist = np.append(1,SHshlist)
        biotlist = np.transpose(formlist)[15]
        biotlist = np.append(1,biotlist)
        
        alphalist = alphalist.astype(float)
        betalist = betalist.astype(float)
        gammalist = gammalist.astype(float)
        tecBlist = tecBlist.astype(float)
        SHshlist = SHshlist.astype(float)
        biotlist = biotlist.astype(float)
        print("TecFackist: ",tecBlist)
        print(alphalist,ftvdlist)
        i=0
        while i<len(fttvdlist):
            betalist[i],gammalist[i] = getEuler(float(alphalist[i]),float(betalist[i]),float(gammalist[i]))
            difftvd[i] = float(fttvdlist[i])-float(ttvdlist[i])
            fttvdlist[i] = float(fttvdlist[i])
            ttvdlist[i] = float(ttvdlist[i])
            hydrldiff[i]= float(difftvd[i])
            if hydrldiff[i]<0:
                hydrldiff[i]=0
            htvdlist[i] = float(fttvdlist[i])+float(hydrldiff[i])
            i+=1
        difftvd = np.append(difftvd,difftvd[-1])
        hydrldiff = np.append(hydrldiff,hydrldiff[-1])
        fttvdlist = np.append(fttvdlist,final_depth)
        ttvdl = float(fttvdlist[-1])-float(difftvd[-1])
        htvdl = float(fttvdlist[-1])-float(hydrldiff[-1])
        ttvdlist = np.append(ttvdlist,ttvdl)
        htvdlist = np.append(htvdlist,htvdl)
        
        fttvdlist=fttvdlist.astype(float)
        logtoplist=logtoplist.astype(float)
        strucbotlist=strucbotlist.astype(float)
        ttvdlist=ttvdlist.astype(float)
        logbotlist=logbotlist.astype(float)
        htvdlist=htvdlist.astype(float)
        print("Differential TVD list:")
        print([difftvd,hydrldiff,fttvdlist,ttvdlist])
        structoplist = np.delete(ttvdlist,-1)
        goclist = np.where(goclist.astype(float) == 0, np.nan, goclist.astype(float))
        owclist = np.where(owclist.astype(float) == 0, np.nan, owclist.astype(float))
        cdtvdlist = structoplist+((strucbotlist-structoplist)*centroid_ratio_list)
        print("Structural tops list",structoplist)
        print("Structural bottoms list",strucbotlist)
        print("Structural centroids ratios",centroid_ratio_list)
        print("Structural centroids list",cdtvdlist)
        print("tops list",logtoplist)
        print("bottoms list",logbotlist)
        print("OWCs",owclist)
        print("GOCs",goclist)
        print("GR Cutoffs",grlist)
    
    j=0
    k=0
    m=0
    n=0
    o=0
    p=0
    i=0
    nu3 = [nu] * (len(tvd))
    mu2 = [0.6] * (len(tvd))
    ucs2 = [np.nan] * (len(tvd))
    try:
        print(lithot)
    except:
        pass
    while(i<len(tvd)):
        if tvdbgl[i]<0:
            tvdbglf[i] = 0
            if tvdmsl[i]>0:
                wdfi[i]=tvdmsl[i]*3.2804
        else:
            tvdbglf[i] = tvdbgl[i]*3.28084
        if tvdmsl[i]<0:
            tvdmslf[i] = 0
        else:
            tvdmslf[i] = tvdmsl[i]*3.28084
        
        if md[i]<mud_weight[j][1]:
            mudweight[i] = mud_weight[j][0]
        else:
            mudweight[i] = mud_weight[j][0]
            j+=1
        if md[i]<tgrads[o][1]:
            y = [bht_point[o-1][0],bht_point[o][0]]
            x = [tvdbgl[find_nearest_depth(md,bht_point[o-1][1])[0]],tvdbgl[find_nearest_depth(md,bht_point[o][1])[0]]]
            tempC[i] = np.interp(tvdbgl[i],x,y)#/1000
            y2 = [tgrads[o-1][0],tgrads[o][0]]
            x2 = [tvdbgl[find_nearest_depth(md,tgrads[o-1][1])[0]],tvdbgl[find_nearest_depth(md,tgrads[o][1])[0]]]
            tempG[i] = np.interp(tvdbgl[i],x2,y2)/1000
            delTempC[i] = tempC[i]-mudtemp
        else:
            tempG[i] = tgrads[o][0]/1000
            tempC[i] = bht_point[o][0]
            delTempC[i] = mudtemp-tempC[i]
            o+=1
        if lithos is not None:
            if md[i]<lithot[k][0]:
                lithotype[i] = int(lithot[k-1][1])
                if len(lithot[k])>2 and lithot[k-1][2]>0:
                    try:
                        nu2[i] = lithot[k-1][2]
                    except:
                        pass
                    try:
                        if(lithot[k-1][3])>0:
                            mu2[i] = lithot[k-1][3]
                    except:
                        pass
                    try:
                        ucs2[i] = lithot[k-1][4]
                    except:
                        pass
                else:
                    numodel = int(lithotype[i]) + 16
                    nu2[i] = float(model[numodel])
            else:
                lithotype[i] = int(lithot[k][1])
                try:
                    nu2[i] = float(lithot[k][2])
                except:
                    pass
                try:
                    if(lithot[k][3])>0:
                        mu2[i] = float(lithot[k][3])
                except:
                    pass
                try:
                    ucs2[i] = float(lithot[k][4])
                except:
                    pass
                k+=1
        if flags is not None:
            if md[i]<imagelog[m][0]:
                ilog[i] = int(imagelog[m-1][1])
            else:
                ilog[i] = imagelog[m][1]
                m+=1
        if forms is not None:
            formtvd[i] = np.interp(tvd[i],fttvdlist,ttvdlist)
            btvd2[i] = np.interp(tvd[i],logtoplist,logbotlist)
            hydrotvd[i] = np.interp(tvd[i],fttvdlist,htvdlist)
            #cdtvd1[i] = np.interp(tvd[i],logtoplist,cdtvdlist)
            if tvd[i]<=float(ftvdlist[p]):
                #cdtvd2[i] = cdtvdlist[p]
                alphas[i] = alphalist[p] if np.isfinite(alphalist[p]) else offset
                betas[i] = betalist[p] if np.isfinite(betalist[p]) else strike
                gammas[i] = gammalist[p] if np.isfinite(gammalist[p]) else dip
                tecB[i] = tecBlist[p] if np.isfinite(tecBlist[p]) else tecb
                SHsh[i] = SHshlist[p] 
                biot[i] = biotlist[p] if np.isfinite(biotlist[p]) else 1
                grcut[i] = grlist[p]
                ttvd[i] = logtoplist[p]
                btvd[i] = logbotlist[p]
                Owc[i]=owclist[p]
                Goc[i]=goclist[p]
                structop[i] =ttvdlist[p]
                strucbot[i] =strucbotlist[p]
                hydroheight[i] = tvd[i]+hydrldiff[p]
                if np.isfinite(float(btlist[p])):
                    bt[i] = float(btlist[p])
                else:
                    bt[i] = 0
            else:
                #cdtvd2[i] = cdtvdlist[p]
                grcut[i] = grlist[p]
                ttvd[i] = logtoplist[p]
                btvd[i] = logbotlist[p]
                Owc[i]=np.nan
                Goc[i]=np.nan
                structop[i] =structoplist[p]
                strucbot[i] =strucbotlist[p]
                hydroheight[i] = tvd[i]+hydrldiff[p]
                if np.isfinite(float(btlist[p])):
                    bt[i] = float(btlist[p])
                else:
                    bt[i] = 0
                p+=1
        else:
            grcut[i] = np.nanmean(gr)
        i+=1
    #cdtvd = (structop+strucbot)/2
    #tempG[:] = tempG[:]*1000

    #check plot
    """plt.plot(lithotype,md)
    plt.plot(nu3,md)
    plt.plot(nu2,md)
    plt.plot(mu2,md)
    plt.plot(ucs2,md)
    plt.plot(ilog,md)
    plt.show()
    plt.clf()"""
    #plt.plot(formtvd,tvd)
    if forms is not None:
        try:
            plt.plot(structop, tvd, label="Structural Tops", linestyle=':')
            plt.plot(strucbot, tvd, label="Structural Bottoms", linestyle=':')
            plt.plot(Owc, tvd, label="OWC", linestyle='-')
            plt.plot(Goc, tvd, label="GOC", linestyle='-')
            plt.plot(btvd, tvd, label="Log Bottom", linestyle='-')
            plt.plot(ttvd, tvd, label="Log Top", linestyle='-')
            
            plt.gca().invert_yaxis()
            plt.title(well.name + well.uwi + " Structure Diagram ")
            plt.legend(loc='upper right')
            #plt.show()
            plt.savefig(os.path.join(output_dir, "Structure.png"))
            plt.close()
        except:
            pass
    
    lradiff = np.full(len(md),np.nan)
    
    if alias['resshal'] != [] and alias['resdeep'] != []:
        rS = well.data[alias['resshal'][0]].values
        rD = well.data[alias['resdeep'][0]].values
        print(rD)
        #sd = gr.plot_2d(cmap='viridis_r', curve=True, lw=0.3, edgecolor='k')
        #sd2 = kth.plot_2d(cmap='viridis_r', curve=True, lw=0.3, edgecolor='k')
        #plt.xlim(0,150)
        rdiff = rD[:]-rS[:]
        rdiff[np.isnan(rdiff)] = 0
        #rdiff = interpolate_nan(rdiff,0)
        radiff = (rdiff[:]*rdiff[:])**0.5
        #plt.plot(radiff)
        #plt.yscale('log')
        #i = 0
        lradiff = radiff
        #while i<len(radiff):
        #    lradiff[i] = radiff[i]
        #    i+=1
        #print("Rdiff :",lradiff)
    
        rediff = Curve(lradiff, mnemonic='ResD',units='ohm/m', index=md, null=0)
        well.data['ResDif'] =  rediff
        print("sfs = :",sfs)
        if forms is not None:
            shaleflag = np.where(gr < grcut, 1, 0)
            zoneflag = np.where(gr < grcut, 0, 1)
        else:
            shaleflag = rediff.block(cutoffs=sfs,values=(0,1)).values
            zoneflag = rediff.block(cutoffs=sfs,values=(1,0)).values
        print(shaleflag)
        #plt.plot(shaleflag,tvd)        
        #plt.show()
        #plt.close()
    else:
        shaleflag = np.zeros(len(md))
        zoneflag = np.zeros(len(md))
    shaleflagN = (np.max(shaleflag)-shaleflag[:])/np.max(shaleflag)
    flag = Curve(shaleflag, mnemonic='ShaleFlag',units='ohm/m', index=md, null=0)
    zone = Curve(zoneflag, mnemonic='ShaleFlag',units='ohm/m', index=md, null=0)
    well.data['Flag'] =  flag
    
    #print(shaleflag)
    
    print("air gap is ",agf,"feet")
    if glwd>=0:
        print("Ground Level is ",glf,"feet above MSL")
    if glwd<0:
        print("Seafloor is ",wdf,"feet below MSL")
        print(wdfi)
        
    ##print(attrib[1])
    
    rhoppg = np.zeros(len(tvdf))
    rhogcc = np.zeros(len(tvdf))
    ObgTppg = np.zeros(len(tvdf))
    hydrostatic = np.zeros(len(tvd))
    mudhydrostatic = np.zeros(len(tvd))
    lithostatic = np.zeros(len(tvd))
    i = 1
    #while(i<len(tvdf-1)):
    #   rhoppg[i] = rhoappg +(((tvdf[i]-agf-wdf)/3125)**a) #amoco formula for density
    #    i+=1
    while(i<len(tvdf-1)):
        if glwd<0:
            if(tvdbgl[i]>=0):
                rhoppg[i] = rhoappg +(((tvdf[i]-agf-wdf)/3125)**a)
                rhogcc[i] =  0.11982642731*rhoppg[i]
                if np.isfinite(zden2[i]):
                    if zden2[i]<4:
                        rhoppg[i] = zden2[i]/0.11982642731
                        rhogcc[i] = zden2[i]
                hydrostatic[i] = water
                mudhydrostatic[i] = 1.0*mudweight[i]
            else:
                if(tvdmsl[i]<0):
                    rhoppg[i] = 8.34540426515252*water
                    rhogcc[i] =  0.11982642731*rhoppg[i]
                    hydrostatic[i] = 0
                    mudhydrostatic[i] = 0
                else:
                    rhoppg[i] = 0
                    rhogcc[i] = 0
                    hydrostatic[i] = water
                    mudhydrostatic[i] = 1.0*mudweight[i]
        else:
            if(tvdbgl[i]>=0):
                rhoppg[i] = rhoappg +(((tvdbglf[i])/3125)**a)
                rhogcc[i] =  0.11982642731*rhoppg[i]
                if np.isfinite(zden2[i]):
                    if zden2[i]<4:
                        rhoppg[i] = zden2[i]/0.11982642731
                        rhogcc[i] = zden2[i]
                hydrostatic[i]= water
                mudhydrostatic[i] = 1.0*mudweight[i]
            else:
                rhoppg[i] = 0
                rhogcc[i] = 0
                hydrostatic[i] = 0
                mudhydrostatic[i] = 0
        i+=1
    #hydrostatic =  (water*9.80665/6.89476) * tvdmsl
    hydroppf = 0.4335275040012*hydrostatic
    mudppf = 0.4335275040012*mudhydrostatic
    lithostatic =  (2.6*9.80665/6.89476) * tvd
    gradient = lithostatic/(tvdf)*1.48816
    rhoppg[0] = rhoappg
    rhogcc[0] = rhoappg*0.11982642731
    try:
        rhogcc = [rhogcc[i] if math.isnan(zden2[i]) else zden2[i] for i in range(len(zden2))]
    except:
        pass
    #rhoppg = interpolate_nan(rhoppg)
    
    #rhogcc[0] = 0.01
    
    integrho = np.zeros(len(tvd))
    integrhopsift = np.zeros(len(tvd))
    i=1
    maxwaterppg = wdf*8.34540426515252*water
    while(i<len(tvdf-1)):
        if glwd<0:
            if(tvdbgl[i]>0):
                if(tvdmsl[i]>0):
                    #integrho[i] = integrho[i-1]+(rhogcc[i-1]*dtvd[i-1])
                    integrho[i] = integrho[i-1]+(rhogcc[i]*9806.65*(tvdbgl[i]-tvdbgl[i-1])) #in pascals
                    integrhopsift[i] = (integrho[i]*0.000145038)/tvdf[i]
                    ObgTppg[i] =((maxwaterppg + ((np.mean(rhoppg[i]))*(tvdbglf[i])))/tvdmslf[i])
            else:
                if(tvdmsl[i]>0):
                    integrho[i] = integrho[i-1]+(water*9806.65*(tvdbgl[i]-tvdbgl[i-1])) #in pascals
                    integrhopsift[i] = (integrho[i]*0.000145038)/tvdf[i]
                    ObgTppg[i] =(8.34540426515252*water)
        else:
            if (tvdbgl[i]>0):
                integrho[i] = integrho[i-1]+(rhogcc[i]*9806.65*(tvdbgl[i]-tvdbgl[i-1])) #in pascals
                integrhopsift[i] = (integrho[i]*0.000145038)/tvdf[i]
                ObgTppg[i] =((np.mean(rhoppg[i]))*(tvdbglf[i]))/tvdf[i] #Curved Top Obg Gradient
                #ObgTppg[i] =rhoppg[i] #Flat Top Obg Gradient
        i+=1
     
    #ObgTppg = integrhopsift*19.25
    ObgTgcc = 0.11982642731*ObgTppg
    ObgTppf = 0.4335275040012*ObgTgcc
    ObgTgcc[0] = 0.01
    print("Obg: ",ObgTgcc)
    print("len of Obg: ",len(ObgTgcc))
    print("Zden: ",zden2)
    print("len of zden: ",len(zden2))
    import math
    coalflag = np.zeros(len(tvd))
    lithoflag = np.zeros(len(tvd))
    try:
        ObgTgcc = [ObgTgcc[i] if math.isnan(zden2[i]) else zden2[i] for i in range(len(zden2))]
        coalflag = [0 if math.isnan(zden2[i]) else 1 if zden2[i]<1.6 else 0 for i in range(len(zden2))]
        lithoflag = [0 if shaleflag[i]<1 else 1 if zden2[i]<1.6 else 2 for i in range(len(zden2))]
    except:
        pass
    
    coal = Curve(lithotype, mnemonic='CoalFlag',units='coal', index=tvd, null=0)
    litho = Curve(lithoflag, mnemonic='LithoFlag',units='lith', index=tvd, null=0)
    
    #zhangs

    ct = 0 #delta m/s per metre of depth
    ct = ct*0.1524
    ct = lamb #zhang's value for gulf of mehico
    pn = water #Hydrostatic, in gcc
    
    matrick = dtmt #us/ft
    mudline = dtml #us/ft
    print("HUZZAH")
    dalm = dt.as_numpy()*1
    tvdm = well.data['TVDM'].as_numpy()*1
    tvdm[0] = 0.1
    print("TVDM",tvdm)
    #dalm = 1000000/dalm
    matrix = np.zeros(len(dalm))
    
    i=0
    while i<(len(dalm)):
        matrix[i] = matrick + (ct*i)
        if lithotype[i]>1.5:
            matrix[i] = 65
        if tvdbgl[i]>0:
            if(dalm[i]<matrick):
                dalm[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
            if(np.isnan(dalm[i])):
                dalm[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
        
        i+=1
    import math
    print(dalm)
    
    vpinv = (dalm*(10**-6)*3280.84)
    vp = 1/vpinv  #km/s
    print(vp)
    
    if glwd<0:
        hydropsi = hydroppf[:]*(tvd[:]*3.28084)#tvdmslf[:]
        obgpsi= integrho*0.000145038
        #obgpsi = np.array([np.mean(ObgTppf[0:i]) * tvdmslf[i-1] for i in range(1, len(ObgTppf) + 1)])
    else:
        hydropsi = hydroppf[:]*(tvd[:]*3.28084)#tvdbglf[:]
        #obgpsi = np.array([np.mean(ObgTppf[0:i]) * tvdbglf[i-1] for i in range(1, len(ObgTppf) + 1)])
        obgpsi= integrho*0.000145038
    mudpsi = mudppf[:]*tvdf[:]
    i = 0
    ppgZhang = np.zeros(len(tvdf))
    gccZhang = np.zeros(len(tvdf))
    psiZhang = np.zeros(len(tvdf))
    psiZhang2 = np.zeros(len(tvdf))
    psiftZhang = np.zeros(len(tvdf))
    psiftZhang2 = np.zeros(len(tvdf))
    gccZhang2 = np.zeros(len(tvdf))
    pnpsi = np.zeros(len(tvdf))
    psipp = np.zeros(len(tvdf))
    psiftpp = np.zeros(len(tvdf))
    horsud = np.zeros(len(tvdf))
    lal = np.zeros(len(tvdf))
    ym = np.zeros(len(tvdf))
    sm = np.zeros(len(tvdf))
    bm = np.zeros(len(tvdf))
    cm_sip = np.zeros(len(tvdf))
    lal3 = np.zeros(len(tvdf))
    phi = np.zeros(len(tvdf))
    philang = np.zeros(len(tvdf))
    H = np.zeros(len(tvdf))
    K = np.zeros(len(tvdf))
    dtNormal = np.zeros(len(tvdf))
    dtNormal[:] = matrick
    
    #ObgTppg[0] = np.nan
    print("ObgTppg:",ObgTppg)
    print("Reject Subhydrostatic below ",underbalancereject)

    if UCSs is not None:
        ucss = UCSs.to_numpy()
    print("Lithos: ",lithos)
    print("UCS: ",UCSs)
    print("IMAGE: ",flags)
    
    #b=3.001 #b>c
    #c=3 #b is the compaction constant of the unloading case and c is the compaction constant of the loading case
    #c = ct/tvdbgl[i] or c = ct/tvdbglf[i]
    #deltmu0 = 70
    maxveldepth = ul_depth
    if ul_depth == 0:
        mvindex = np.nan
    else:
        mvindex = find_nearest_depth(tvd,maxveldepth)[0]
    deltmu0 = np.nanmean(dalm[(find_nearest_depth(tvd,maxveldepth)[0]-5):(find_nearest_depth(tvd,maxveldepth)[0]+5)])
    c=ct
    b=ct
    print("Max velocity is ",deltmu0,"uspf")
    while i<(len(ObgTppg)-1):
        if glwd>=0: #Onshore Cases
            if tvd[i]>ul_depth:
                b = ul_exp
            if tvdbgl[i]>0:
                if shaleflag[i]<0.5: #Shale PorePressure
                    gccZhang2[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    gccZhang[i] = (ObgTgcc[i] - ((ObgTgcc[i]-(pn*1))/(b*tvdbgl[i]))*((((b-c)/c)*(math.log((mudline-matrick)/(deltmu0-matrick))))+(math.log((mudline-matrick)/(dalm[i]-matrick)))))/1
                else:
                    gccZhang[i] = np.nan #Hydraulic Pore Pressure
                    gccZhang2[i] = np.nan
                if gccZhang[i]<underbalancereject: #underbalance reject
                    gccZhang[i]=underbalancereject
                if gccZhang2[i]<underbalancereject:
                    gccZhang2[i]=underbalancereject
                
                gccZhang[np.isnan(gccZhang)] = gccZhang2[np.isnan(gccZhang)]
                ppgZhang[i] = gccZhang[i]*8.33
                dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                lal3[i] = lall*(304.8/(dalm[i]-1))
                lal[i] = lalm*(vp[i]+lala)/(vp[i]**lale) #S0, cohesive strength, in MPa, for all rocks, for gassy rocks apply gassman correction
                horsud[i] = horsuda*(vp[i]**horsude)
                if np.isnan(ucs2[i]) or ucs2[i]==0:
                    ucs2[i] = horsud[i]
                phi[i] = np.arcsin(1-(2*nu2[i]))
                philang[i] = np.arcsin((vp[i]-1.5)/(vp[i]+1.5))
                H[i] = (4*(np.tan(phi[i])**2))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i]))))
                K[i] = (4*lal[i]*(np.tan(phi[i])))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i])))) 
                ym[i] = 0.076*(vp[i]**3.73)*(1000) #in GPa
                sm[i] = 0.03*(vp[i]**3.30) #in GPa
                bm[i] = ym[i]/(3*(1-(2*nu2[i]))) #same units as ym
                psiftpp[i] = 0.4335275040012*gccZhang[i]
                psipp[i] = psiftpp[i]*tvdf[i]
                #if psipp[i]<hydropsi[i]:
                #   psipp[i] = hydropsi[i]
        else: #Offshore Cases
            if tvd[i]>ul_depth:
                b = ul_exp
            if tvdbgl[i]>0:
                if shaleflag[i]<0.5:#Shale Pore Pressure
                    gccZhang2[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    gccZhang[i] = (ObgTgcc[i] - ((ObgTgcc[i]-(pn*1))/(b*tvdbgl[i]))*((((b-c)/c)*(math.log((mudline-matrick)/(deltmu0-matrick))))+(math.log((mudline-matrick)/(dalm[i]-matrick)))))/1
                    #gccZhang[i] = getGccZhang(ObgTgcc[i],pn,mudline,matrick,dalm[i],ct,tvdbgl[i])
                else:
                    gccZhang[i] = np.nan #Hydraulic Pore Pressure
                    gccZhang2[i] = np.nan
                if gccZhang[i]<underbalancereject: #underbalance reject
                    gccZhang[i]=underbalancereject
                if gccZhang2[i]<underbalancereject:
                    gccZhang2[i]=underbalancereject
                
                gccZhang[np.isnan(gccZhang)] = gccZhang2[np.isnan(gccZhang)]
                ppgZhang[i] = gccZhang[i]*8.33
                dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                lal3[i] = lall*(304.8/(dalm[i]-1))
                lal[i] = lalm*(vp[i]+lala)/(vp[i]**lale)
                horsud[i] = horsuda*(vp[i]**horsude)
                if np.isnan(ucs2[i]) or ucs2[i]==0:
                    ucs2[i] = horsud[i]
                phi[i] = np.arcsin(1-(2*nu2[i]))
                philang[i] = np.arcsin((vp[i]-1)/(vp[i]+1))
                H[i] = (4*(np.tan(phi[i])**2))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i]))))
                K[i] = (4*lal[i]*(np.tan(phi[i])))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i])))) 
                ym[i] = 0.076*(vp[i]**3.73)*(1000) #in GPa
                sm[i] = 0.03*(vp[i]**3.30) #in GPa
                bm[i] = ym[i]/(3*(1-(2*nu2[i]))) #same units as ym
                psiftpp[i] = 0.4335275040012*gccZhang[i]
                psipp[i] = psiftpp[i]*tvdf[i]
        i+=1
    
    """plt.plot(phi,tvd, label='nu')
    plt.plot(philang,tvd, label='lang')
    plt.plot(philang-phi,tvd, label='delta')
    plt.legend()
    plt.show()
    plt.close()"""
    #gccZhang[0] = np.nan
    dphi = np.degrees(phi[:])
    gccZhang[-1] = hydrostatic[-1]
    gccZhang[0] = hydrostatic[0]
    gccZhang[np.isnan(gccZhang)] = water
    #gccZhang2[-1] = hydrostatic[-1]
    #gccZhang2[0] = hydrostatic[0]
    #gccZhang2[np.isnan(gccZhang2)] = water
    ppgZhang[np.isnan(ppgZhang)] = water*8.33
    psiftpp = interpolate_nan(psiftpp)
    psipp = interpolate_nan(psipp)
    print("GCCZhang: ",gccZhang)
    psiftpp = 0.4335275040012*gccZhang
    
    """
    #Check Plot
    plt.plot(gccZhang,tvd, label='Unloading')
    plt.plot(gccZhang2,tvd, label='Loading',alpha=0.5, linestyle='-')
    plt.legend(loc="upper left")
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    """
    #psipp = psiftpp[:]*tvdf[:]
    #ObgTgcc = np.array(ObgTgcc)
    #obgpsift = 0.4335275040012*ObgTgcc
    #plt.plot(cm_sip,md, alpha=0.5)
    #plt.show()
    #plt.plot(ucs2,md)
    #plt.plot(dtNormal)
    #plt.plot(dalm)
    ###plt.show()
    #plt.clf()
    """plt.close()
    plt.plot(ym,tvd)
    plt.show()
    plt.clf()
    plt.close()"""
    
    
    #Eatons/Daines
    print("Tectonic factor input = ",tecb)
    i = 0
    mu = 0.65
    if b > 10.0:
        b=0
    fgppg = np.zeros(len(ppgZhang))
    fgcc = np.zeros(len(ppgZhang))
    mufgppg = np.zeros(len(ppgZhang))
    mufgcc = np.zeros(len(ppgZhang))

    while i<(len(ObgTppg)-1):
        if tvdbgl[i]>0:
            if shaleflag[i]<0.5:
                fgppg[i] = (nu2[i]/(1-nu2[i]))*(ObgTppg[i]-(biot[i]*ppgZhang[i]))+(biot[i]*ppgZhang[i]) +(tecB[i]*(ObgTppg[i]))
                mufgppg[i] = ((1/((((mu**2)+1)**0.5)+mu)**2)*(ObgTppg[i]-ppgZhang[i])) + ppgZhang[i]
                mufgcc[i] = 0.11982642731*mufgppg[i]
            else:
                fgppg[i] = np.nan
                mufgppg[i] = np.nan
                mufgcc[i] = np.nan
        fgcc[i] = 0.11982642731*fgppg[i]
        i+=1
    fgppg = interpolate_nan(fgppg)
    fgcc = interpolate_nan(fgcc)
    #fgppg = (nu/(1-nu))(ObgTppg-ppgZhang)+ppgZhang
    psiftfg = 0.4335275040012*fgcc
    psifg = psiftfg*tvdf
    psimes = ((psifg+obgpsi)/2)+psipp
    shsvratio = psifg/obgpsi
    
    
    if forms is not None:
        psippsand = np.zeros(len(md))
        from hydraulics import getPPfromTopRecursive
        from hydraulics import compute_optimal_gradient
        from hydraulics import getHydrostaticPsi
        gradients = np.zeros(len(md))
        gradlist = np.zeros(len(cdtvdlist))
        eqlithostat = np.zeros(len(md))
        eqlithostat2 = np.zeros(len(md))
        #for j in range(len(strucbotlist)):
        j=0
        #offsets[j] = compute_optimal_offset(tvd[find_nearest_depth(tvd,structop[j])[0]:find_nearest_depth(tvd,strucbot[j])[0]], psipp[find_nearest_depth(tvd,structop[j])[0]:find_nearest_depth(tvd,strucbot[j])[0]], np.nanmean(ObgTgcc[find_nearest_depth(tvd,structop[j])[0]:find_nearest_depth(tvd,strucbot[j])[0]]))
        for i in range(len(md)):
            if structop[i]!=structop[i-1]:
                gradients[i] = compute_optimal_gradient(tvd[find_nearest_depth(tvd,ttvd[i])[0]:find_nearest_depth(tvd,btvd[i])[0]], psipp[find_nearest_depth(tvd,ttvd[i])[0]:find_nearest_depth(tvd,btvd[i])[0]])
                gradlist[j] = gradients[i]
                j+=1
            else:
                gradients[i] = gradients[i-1]
            eqlithostat[i] = getHydrostaticPsi(tvd[i],gradients[i])
            eqlithostat2[i] = getHydrostaticPsi(tvd[i],gradients[i])
            psippsand[i] = getPPfromTopRecursive(0 , shsvratio[find_nearest_depth(tvd,structop[i])[0]], obgpsi[find_nearest_depth(tvd,structop[i])[0]],0.85, water, structop[i], Goc[i], Owc[i], tvd[i])
            #shift[i] = eqlithostat[find_nearest_depth(tvd,cdtvd[i])[0]]-psippsand[find_nearest_depth(tvd,cdtvd[i])[0]]
            #print(shift)
            #psippsand[i]=getPPfromTopRecursive(1000-shift[i], shsvratio[find_nearest_depth(tvd,structop[i])[0]], obgpsi[find_nearest_depth(tvd,structop[i])[0]],0.85, water, structop[i], Goc[i], Owc[i], tvd[i])
            #print(i)
        
        shalepressures = np.zeros((len(cdtvdlist),len(md)))
        shifts = np.zeros((len(cdtvdlist),len(md)))
        for i, depth in enumerate(cdtvdlist):
            shalepressures[i] = getHydrostaticPsi(tvd,gradlist[i])
            
        centroid_pressures_sand = np.zeros(len(cdtvdlist))
        # Find the nearest pressure for each depth in cdtvdlist
        for i, depth in enumerate(cdtvdlist):
            nearest_idx = find_nearest_depth(tvd, depth)[0]
            centroid_pressures_sand[i] = psippsand[nearest_idx]
        
        centroid_pressures_shale = np.zeros(len(cdtvdlist))
        # Find the nearest pressure for each depth in cdtvdlist
        for i, depth in enumerate(cdtvdlist):
            nearest_idx = find_nearest_depth(tvd, depth)[0]
            centroid_pressures_shale[i] = shalepressures[i][nearest_idx]
            #if tvd[i]<logbotlist[j]:
                #j+=1
        
        shifts = centroid_pressures_sand - centroid_pressures_shale
        print("centroid pressure hydrostatic: ",centroid_pressures_sand)
        print("centroid pressure in shale: ",centroid_pressures_shale)
        print("Max seal integrity pressure: ",shifts)
        
        j = 0
        for i in range(len(md)):
            try:
                if tvd[i]<logbotlist[j]:
                    psippsand[i] = psippsand[i]-shifts[j]
                else:
                    j+=1
                    psippsand[i] = np.nan#psippsand[i]-shifts[j-1]
            except:
                pass
    
        for i in range(len(md)):
            if shaleflag[i]>0.5:
                psipp[i] = psippsand[i]
        psipp = interpolate_nan(psipp)
        #psippsand = interpolate_nan(psippsand)
        #from DrawSP import getSHMax_optimized
        
            #psisfl = (psimes[:]*H[:])+K[:]
        shsvratio2 = psifg/obgpsi
        
        """
        plt.plot(psipp,tvd)
        plt.plot(eqlithostat,tvd)
        plt.plot(psippsand,tvd)
        plt.plot(obgpsi,tvd)
        plt.plot(psifg,tvd)
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()
        
        plt.plot(psipp/tvdf,tvd)
        plt.plot(psippsand/tvdf,tvd)
        plt.plot(obgpsi/tvdf,tvd)
        plt.plot(psifg/tvdf,tvd)
        #plt.plot(gradients,tvd)
        plt.gca().invert_yaxis()
        plt.xlim(0.4,1)
        plt.show()
        plt.close()
        """
    
    
    #from DrawSP import getSHMax_optimized
    from DrawSP import getSP
    from DrawSP import drawSP
    
    i=0
    sgHMpsi = np.zeros(len(tvd))
    sgHMpsiL = np.zeros(len(tvd))
    sgHMpsiU = np.zeros(len(tvd))
    psisfl = np.zeros(len(tvd))
    while i<len(tvd)-1:
        try:
            stresshratio = SHsh[i]
        except:
            stresshratio = np.nan
        if np.isfinite(stresshratio):
            sgHMpsi[i] = psifg[i]*stresshratio
            sgHMpsiL[i] = sgHMpsi[i]*0.9 #10% margin of uncertainity
            sgHMpsiU[i] = sgHMpsi[i]*1.1 #10% margin of uncertainity
        else:
            result = getSP(obgpsi[i]/145.038,psipp[i]/145.038,mudpsi[i]/145.038,psifg[i]/145.038,ucs2[i],phi[i],ilog[i],mu2[i],nu2[i],bt[i],ym[i],delTempC[i])
            sgHMpsi[i] = (result[2])*145.038
            sgHMpsiL[i] = (result[0])*145.038
            sgHMpsiU[i] = (result[1])*145.038
        if psifg[i]<obgpsi[i]:#in normal and strikeslip regimes
            psifg[i] = np.nanmin([psifg[i],sgHMpsiL[i]])
        #psisfl[i] = 0.5*((3*sgHMpsi[i])-psifg[i])*(1-np.sin(np.radians(phi[i]))) -(horsud[i]*145.038/10*np.cos(np.radians(phi[i])))+ (psipp[i]*np.sin(np.radians(phi[i])))
        i+=1
    sgHMpsi = interpolate_nan(sgHMpsi)
    #psisfl = (psimes[:]*H[:])+K[:]
    
    from BoreStab import get_optimal
    from BoreStab import draw
    
    
    

    #params = {'mnemonic': 'AMOCO', 'run':0, }
    #params={'AMOCO': {'units': 'G/C3'}}
    #data = rhogcc
    #ObgTgcc[0] = np.nan
    
    
    
    
    
    
    #FILTERS
    #Smooth curves using moving averages
    i = window
    sfg = fgcc
    spp = gccZhang
    spsipp = psipp
    shorsud = horsud
    slal = lal
    slal2 = ym
    slal3 = sm
    spsifp = psifg
    ssgHMpsi = sgHMpsi
    ssgHMpsiL = sgHMpsiL
    ssgHMpsiU = sgHMpsiU
    while i<len(fgcc):
        sum1 = np.sum(gccZhang[(i-window):i+(window)])
        spp[i] = sum1/(2*window)
        sum2 = np.sum(psipp[(i-window):i+(window)])
        spsipp[i] = sum2/(2*window)
        sum3 = np.sum(psipp[(i-window):i+(window)])
        spsipp[i] = sum3/(2*window)
        sum4 = np.sum(ucs2[(i-window):i+(window)])
        shorsud[i] = sum4/(2*window)
        sum5 = np.sum(lal[(i-window):i+(window)])
        slal[i] = sum5/(2*window)
        sum6 = np.sum(ym[(i-window):i+(window)])
        slal2[i] = sum6/(2*window)
        sum7 = np.sum(sm[(i-window):i+(window)])
        slal3[i] = sum7/(2*window)
        sum8 = np.sum(psifg[(i-window):i+(window)])
        spsifp[i] = sum8/(2*window)
        sum9 = np.sum(fgcc[(i-window):i+(window)])
        sfg[i] = sum9/(2*window)
        sum10 = np.sum(sgHMpsi[(i-window):i+(window)])
        ssgHMpsi[i] = sum10/(2*window)
        sum11 = np.sum(sgHMpsiL[(i-window):i+(window)])
        ssgHMpsiL[i] = sum11/(2*window)
        sum12 = np.sum(sgHMpsiU[(i-window):i+(window)])
        ssgHMpsiU[i] = sum12/(2*window)
        i+=1
    finaldepth = find_nearest_depth(tvdm,finaldepth)[1]
    doi = min(doi,finaldepth-1)
    if doi>0:
        doiactual = find_nearest_depth(tvdm,doi)
        print(doiactual)
        doiA = doiactual[1]
        doiX = doiactual[0]
        print("Depth of interest :",doiA," with index of ",doiX)
        devdoi = well.location.deviation[doiX]
        incdoi = devdoi[2]
        azmdoi = devdoi[1]
        print("Inclination is :",incdoi," towards azimuth of ",azmdoi)
        sigmaVmpa = obgpsi[doiX]/145.038
        sigmahminmpa = spsifp[doiX]/145.038
        ppmpa = spsipp[doiX]/145.038
        bhpmpa = mudpsi[doiX]/145.038
        ucsmpa = shorsud[doiX]
        ilog_flag=ilog[doiX]
        print("nu is ",nu2[doiX])
        print("phi is ",np.degrees(phi[doiX]))
        drawSP(output_fileSP,sigmaVmpa,ppmpa,bhpmpa,sigmahminmpa,ucsmpa,phi[doiX],ilog_flag,mu2[doiX],nu2[doiX],bt[doiX],ym[doiX],delTempC[doiX])
        sigmaHMaxmpa = sgHMpsi[doiX]/145.038
        print("SigmaHM = ",sigmaHMaxmpa)
        sigmas = [sigmaHMaxmpa,sigmahminmpa,sigmaVmpa]
        print(sigmas)
        
        """
        if sigmas[2]>sigmas[0]:
            alpha = 0
            beta = 90 #normal faulting regime
            gamma = 0
            print("normal")
        else:
            if(sigmas[2]<sigmas[1]):
                alpha = 0
                beta = 0 #reverse faulting regime
                gamma = 0
                print("reverse")                  
            else:
                alpha = 0 #strike slip faulting regime
                beta = 0
                gamma = 90
                print("Strike slip")
        
        """
        
        alpha = offset
        beta= tilt
        gamma= tiltgamma
        from BoreStab import getRota
        Rmat = getRota(alphas[doiX],betas[doiX],gammas[doiX])
        #sigmas[2] = sigmaVmpa+(sigmaVmpa - sigmaVmpa*Rmat[2][2])/Rmat[2][2]
        print(sigmas)
        #sigmas.sort(reverse=True)
        sigmas.append(bhpmpa-ppmpa)
        sigmas.append(ppmpa)
        
        from PlotVec import savevec
        from PlotVec import showvec
        from BoreStab import getStens
        print("Actual Sv is ",sigmas[2],"Mpa")
        m = np.min([sigmas[0],sigmas[1],sigmas[2]])
        osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alphas[doiX],betas[doiX],gammas[doiX])
        sten = getStens(osx,osy,osz,alphas[doiX],betas[doiX],gammas[doiX])
        sn,se,sd = np.linalg.eigh(sten)[0]
        on,oe,od = np.linalg.eigh(sten)[1]
        savevec(on,oe,od,2,sn,se,sd,output_fileVec)
        #drawStab(sigmas[0],sigmas[1],sigmas[2],sigmas[3],alpha,beta,gamma)
        draw(output_fileS,tvd[doiX],osx,osy,osz,sigmas[3],sigmas[4],ucsmpa,alphas[doiX],betas[doiX],gammas[doiX],0,nu2[doiX],incdoi,azmdoi,bt[doiX],ym[doiX],delTempC[doiX])
        
        #drawDITF(sigmas[0],sigmas[1],sigmas[2],sigmas[3],alpha,beta,gamma)
    
    
    
    TVDF = Curve(tvdf, mnemonic='TVDF',units='m', index=md, null=0)
    TVDMSL = Curve(tvdmsl, mnemonic='TVDMSL',units='m', index=md, null=0)
    TVDBGL = Curve(tvdbgl, mnemonic='TVDMSL',units='m', index=md, null=0)
    TVDM = Curve(tvdm, mnemonic='TVDM',units='m', index=md, null=0)
    
    amoco2 = Curve(rhogcc, mnemonic='RHO',units='G/C3', index=md, null=0)
    well.data['RHOA'] =  amoco2
    
    obgcc = Curve(ObgTgcc, mnemonic='OBG_AMOCO',units='G/C3', index=md, null=0)
    well.data['OBG'] =  obgcc
    
    dtct = Curve(dtNormal, mnemonic='DTCT',units='us/ft', index=md, null=0)
    well.data['DTCT'] =  dtct

    pp = Curve(spp, mnemonic='PP_DT_Zhang',units='G/C3', index=md, null=0)
    well.data['PP'] =  pp
    fg = Curve(sfg, mnemonic='FG_DAINES',units='G/C3', index=md, null=0)
    well.data['FG'] =  fg
    fg2 = Curve(mufgcc, mnemonic='FG_ZOBACK',units='G/C3', index=md, null=0)
    well.data['FG2'] =  fg2
    
    
    pppsi = Curve(spsipp, mnemonic='GEOPRESSURE',units='psi', index=md, null=0, index_name = 'DEPT')
    well.data['PPpsi'] =  pppsi
    fgpsi = Curve(spsifp, mnemonic='FRACTURE_PRESSURE',units='psi', index=md, null=0)
    well.data['FGpsi'] =  fgpsi
    sHMpsi = Curve(ssgHMpsi, mnemonic='SHMAX_PRESSURE',units='psi', index=md, null=0)
    well.data['SHMpsi'] =  sHMpsi
    shmpsi = Curve(ssgHMpsi, mnemonic='shmin_PRESSURE',units='psi', index=md, null=0)
    well.data['shmpsi'] =  shmpsi
    mwpsi = Curve(mudpsi, mnemonic='MUD_PRESSURE',units='psi', index=md, null=0)
    well.data['mwpsi'] =  mwpsi
    mhpsi = Curve(mudweight, mnemonic='MUD_GRADIENT',units='g/cc', index=md, null=0)
    well.data['mhpsi'] =  mhpsi
    c0lalmpa = Curve(slal, mnemonic='C0_Lal',units='MPa', index=md, null=0)
    well.data['C0LAL'] =  c0lalmpa
    c0lal2mpa = Curve(slal2, mnemonic='C0_Lal_Phi',units='MPa', index=md, null=0)
    well.data['C0LAL2'] =  c0lal2mpa
    ucshorsudmpa = Curve(shorsud, mnemonic='UCS_Horsud',units='MPa', index=md, null=0)
    well.data['UCSHORSUD'] =  ucshorsudmpa
    ucslalmpa = Curve(slal3, mnemonic='UCS_Lal',units='MPa', index=md, null=0)
    well.data['UCSLAL'] =  ucslalmpa
    
    #pcal = Curve(psicalib[1], mnemonic='PRESSURE TEST',units='psi', index=psicalib[0], null=0)

    #gcal = Curve(gradcalib[1], mnemonic='BHP GRAD',units='G/C3', index=gradcalib[0], null=0)
    
    #fgcal = Curve(fgradcalib[1], mnemonic='LOT/xLOT/HF GRAD',units='G/C3', index=fgradcalib[0], null=0)
    #mana = len(well._get_curve_mnemonics())
    #print(mana)
    #units = []
    
    output_file4 = os.path.join(output_dir1,"GMech.las")
    output_fileCSV = os.path.join(output_dir1,"GMech.csv")
    df3 = well.df()
    df3.index.name = 'DEPT'
    df3.to_csv(output_fileCSV)
    df3 = df3.reset_index()
    header = well._get_curve_mnemonics()
    lasheader = well.header
    c_units = {"TVDM":"M","RHO":"G/C3", "OBG_AMOCO":"G/C3", "DTCT":"US/F", "PP_DT_Zhang":"G/C3","FG_DAINES":"G/C3","GEOPRESSURE":"PSI","FRACTURE_PRESSURE":"PSI", "SHMAX_PRESSURE":"PSI", "shmin_PRESSURE":"PSI","MUD_PRESSURE":"PSI", "MUD_GRADIENT":"G/C3", "UCS_Horsud":"MPA", "UCS_Lal":"MPA"}
    datasets_to_las(output_file4, {'Header': lasheader,'Curves':df3},c_units)
    #well.to_las('output.las')
    from BoreStab import getHoop, getAlignedStress
    from plotangle import plotfracsQ,plotfrac
    from failure_criteria import plot_sanding
    def drawBHimage(doi):
        hfl = 2.5
        doiactual = find_nearest_depth(tvdm,doi-hfl)
        doiS = doiactual[0]
        doiactual2 = find_nearest_depth(tvdm,doi+hfl)
        doiF = doiactual2[0]
        frac = np.zeros([doiF-doiS,360])
        crush = np.zeros([doiF-doiS,360])
        data=np.zeros([doiF-doiS,4])
        #data2=np.zeros([doiF-doiS,3])
        i=doiS
        j=0
        while i <doiF:
            sigmaVmpa = obgpsi[i]/145.038
            sigmahminmpa = psifg[i]/145.038
            sigmaHMaxmpa = sgHMpsi[i]/145.038
            ppmpa = psipp[i]/145.038
            bhpmpa = mudpsi[i]/145.038
            ucsmpa = horsud[i]
            deltaP = bhpmpa-ppmpa
            sigmas = [sigmaHMaxmpa,sigmahminmpa,sigmaVmpa]
            osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alphas[i],betas[i],gammas[i])
            sigmas = [osx,osy,osz]
            devdoi = well.location.deviation[i]
            incdoi = devdoi[2]
            azmdoi = devdoi[1]
            """
            if sigmas[2]>sigmas[0]:
                alpha = 0
                beta = 90 #normal faulting regime
                gamma = 0
                #print("normal")
            else:
                if(sigmas[2]<sigmas[1]):
                    alpha = 0
                    beta = 0 #reverse faulting regime
                    gamma = 0
                    #print("reverse")                  
                else:
                    alpha = 0 #strike slip faulting regime
                    beta = 0
                    gamma = 90
                    #print("Strike slip")
            sigmas.sort(reverse=True)
            alpha = alpha + offset
            beta= beta+tilt
            """
            cr,fr,minazi,maxazi,minangle,maxangle,angles = getHoop(incdoi,azmdoi,sigmas[0],sigmas[1],sigmas[2],deltaP,ppmpa,ucsmpa,alphas[i],betas[i],gammas[i],nu2[i],bt[i],ym[i],delTempC[i])
            crush[j] = cr
            frac[j] = fr
            if np.max(frac[j])>0:
                data[j] = [tvd[i],minazi,minangle,maxangle]
                #data2[j+1] = [tvd[i+1],round((minazi+180)%360),minangle+180]
            i+=1
            j+=1
        from plotangle import plotfracs, plotfrac
        i=find_nearest_depth(tvdm,doi)[0]
        j=find_nearest_depth(tvdm,doi)[1]
        sigmaVmpa = obgpsi[i]/145.038
        sigmahminmpa = psifg[i]/145.038
        sigmaHMaxmpa = sgHMpsi[i]/145.038
        ppmpa = psipp[i]/145.038
        bhpmpa = mudpsi[i]/145.038
        ucsmpa = horsud[i]
        deltaP = bhpmpa-ppmpa
        sigmas = [sigmaHMaxmpa,sigmahminmpa,sigmaVmpa]
        osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alphas[i],betas[i],gammas[i])
        sigmas = [osx,osy,osz]
        devdoi = well.location.deviation[i]
        incdoi = devdoi[2]
        azmdoi = devdoi[1]
        cr,fr,minazi,maxazi,minangle,maxangle,angles = getHoop(incdoi,azmdoi,sigmas[0],sigmas[1],sigmas[2],deltaP,ppmpa,ucsmpa,alphas[i],betas[i],gammas[i],nu2[i],bt[i],ym[i],delTempC[i])
        fr = np.array(fr)
        angles = np.array(angles)
        data2 = j,fr,angles,minazi,maxazi
        d,f = plotfrac(data2,output_fileFrac)
        plotfracs(data)
        plt.imshow(frac,cmap='Reds',alpha=0.5,extent=[0,360,tvd[doiF],tvd[doiS]],aspect=10)
        plt.imshow(crush,cmap='Blues',alpha=0.5,extent=[0,360,tvd[doiF],tvd[doiS]],aspect=10)
        plt.plot(d, "k-")
        plt.plot(f, "k-",alpha=0.1)
        plt.ylim(j+hfl, j-hfl)
        plt.gca().set_aspect(360/((6.67*hfl*2)*(0.1)))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
        plt.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=True)
        plt.xticks([0,90,180,270,360])
        #plt.grid()
        plt.title("Synthetic Borehole Image")
        plt.savefig(output_fileBHI,dpi=1200)
        plt.clf()
        plt.close()
    
    def plotHoop(doi):
        doiactual = find_nearest_depth(tvdm,doi)
        doiS = doiactual[0]
        i=doiS
        j=0
        sigmaVmpa = obgpsi[i]/145.038
        sigmahminmpa = psifg[i]/145.038
        sigmaHMaxmpa = sgHMpsi[i]/145.038
        ppmpa = psipp[i]/145.038
        bhpmpa = mudpsi[i]/145.038
        ucsmpa = horsud[i]
        deltaP = bhpmpa-ppmpa
        sigmas = [sigmaHMaxmpa,sigmahminmpa,sigmaVmpa]
        osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alphas[i],betas[i],gammas[i])
        sigmas = [osx,osy,osz]
        devdoi = well.location.deviation[i]
        incdoi = devdoi[2]
        azmdoi = devdoi[1]
        getHoop(incdoi,azmdoi,sigmas[0],sigmas[1],sigmas[2],deltaP,ppmpa,ucsmpa,alphas[i],betas[i],gammas[i],nu2[i],bt[i],ym[i],delTempC[i],output_fileHoop)
        
    def drawSand(doi):
        doiactual = find_nearest_depth(tvdm,doi)
        doiS = doiactual[0]
        i=doiS
        j=0
        sigmaVmpa = obgpsi[i]/145.038
        sigmahminmpa = psifg[i]/145.038
        sigmaHMaxmpa = sgHMpsi[i]/145.038
        ppmpa = psipp[i]/145.038
        bhpmpa = mudpsi[i]/145.038
        ucsmpa = horsud[i]
        deltaP = bhpmpa-ppmpa
        sigmas = [sigmaHMaxmpa,sigmahminmpa,sigmaVmpa]
        osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alphas[i],betas[i],gammas[i])
        #sigmas = [osx,osy,osz]
        devdoi = well.location.deviation[i]
        incdoi = devdoi[2]
        azmdoi = devdoi[1]
        Sl = getAlignedStress(osx,osy,osz,alphas[i],betas[i],gammas[i],azmdoi,incdoi)
        sigmamax = max(Sl[0][0],Sl[1][1])
        sigmamin = min(Sl[0][0],Sl[1][1])
        sigma_axial = Sl[2][2]
        k0 = 1
        plot_sanding(os.path.join(output_dir, "Sanding.png"),sigmamax, sigmamin,sigma_axial, ppmpa, ucsmpa, k0, nu2[i])
        #getHoop(incdoi,azmdoi,sigmas[0],sigmas[1],sigmas[2],deltaP,ppmpa,ucsmpa,alpha,beta,gamma,nu2[i],bt[i],ym[i],delTempC[i],output_fileHoop)
        
    def combineHarvest():
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        # Load the images
        image1 = mpimg.imread(output_fileSP)
        image2 = mpimg.imread(output_fileS)
        image3 = mpimg.imread(os.path.join(output_dir, "Sanding.png"))
        image4 = mpimg.imread(output_fileBHI)

        # Create a new figure
        fig, axs = plt.subplots(2, 2,figsize=(16,12))

        # Plot each image in its respective subplot
        axs[0, 0].imshow(image1)
        axs[0, 1].imshow(image2)
        axs[1, 0].imshow(image3)
        axs[1, 1].imshow(image4)
        
        # Remove axes
        for ax in axs.flat:
            ax.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Save the combined image
        plt.savefig(output_fileAll)
        plt.close()
    if doi>0:
        plotHoop(doi)
        drawBHimage(doi)
        drawSand(doi)
        combineHarvest()
    
    from matplotlib.ticker import MultipleLocator
    from Plotter import plot_logs, cutify, cutify2, chopify, choptop

    # Initialize parameters
    tango = min(tango, finaldepth)
    if zulu > finaldepth or zulu > tango:
        zulu = 0

    mogu1 = np.nanmax(ssgHMpsi[:find_nearest_depth(tvd, tango)[0]])#*ureg.psi
    mogu2 = np.nanmax(obgpsi[:find_nearest_depth(tvd, tango)[0]])#*ureg.psi
    mogu3 = np.nanmin(hydropsi[find_nearest_depth(tvd, zulu)[0]:find_nearest_depth(tvd, tango)[0]])#*ureg.psi
    maxchartpressure = 1000*math.ceil(max(mogu1, mogu2)/1000)#*ureg.psi
    minpressure = round(mogu3)#*ureg.psi
    
    Sb = np.full((len(tvd),3,3), np.nan)
    SbFF = np.full((len(tvd),3,3), np.nan)
    hoopmax = np.full(len(tvd),np.nan)
    hoopmin = np.full(len(tvd),np.nan)
    lademax = np.full(len(tvd),np.nan)
    lademin = np.full(len(tvd),np.nan)
    from failure_criteria import mod_lad_cmw, mogi
    print("calculating aligned far field stresses")
    for i in range(0,len(tvd),window):
        #print(i)
        if window>=2.0:
            sigmaVmpa = np.nanmean(obgpsi[i-int(window/2):i+int(window/2)])/145.038
            sigmahminmpa = np.nanmean(psifg[i-int(window/2):i+int(window/2)])/145.038
            sigmaHMaxmpa = np.nanmean(sgHMpsi[i-int(window/2):i+int(window/2)])/145.038
            ppmpa = np.nanmean(psipp[i-int(window/2):i+int(window/2)])/145.038
            bhpmpa = np.nanmean(mudpsi[i-int(window/2):i+int(window/2)])/145.038
            ucsmpa = np.nanmean(horsud[i-int(window/2):i+int(window/2)])
            deltaP = bhpmpa-ppmpa
            sigmas = [sigmaHMaxmpa,sigmahminmpa,sigmaVmpa]
        else:
            sigmaVmpa = np.nanmean(obgpsi[i])/145.038
            sigmahminmpa = np.nanmean(psifg[i])/145.038
            sigmaHMaxmpa = np.nanmean(sgHMpsi[i])/145.038
            ppmpa = np.nanmean(psipp[i])/145.038
            bhpmpa = np.nanmean(mudpsi[i])/145.038
            ucsmpa = np.nanmean(horsud[i])
            deltaP = bhpmpa-ppmpa
            sigmas = [sigmaHMaxmpa,sigmahminmpa,sigmaVmpa]
        try:
            devdoi = well.location.deviation[i]
            incdoi = devdoi[2]
            azmdoi = devdoi[1]
            osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alphas[i],betas[i],gammas[i])
            Sb[i] = getAlignedStress(osx,osy,osz,alphas[i],betas[i],gammas[i],azmdoi,incdoi)
            SbFF[i] = Sb[i]
            Sb[i][0][0] = Sb[i][0][0] - ppmpa
            Sb[i][1][1] = Sb[i][1][1] - ppmpa
            Sb[i][2][2] = Sb[i][2][2] - ppmpa
            sigmaT = (ym[i]*bt[i]*delTempC[i])/(1-nu2[i])
            Szz = np.full(360, np.nan)
            Stt = np.full(360, np.nan)
            Ttz = np.full(360, np.nan)
            Srr = np.full(360, np.nan)
            STMax = np.full(360, np.nan)
            Stmin = np.full(360, np.nan)
            ladempa = np.full(360, np.nan)
            for j in range(0,360,10):
                theta = np.radians(j)
                Szz[j] = Sb[i][2][2] - ((2*nu2[i])*(Sb[i][0][0]-Sb[i][1][1])*(2*math.cos(2*theta))) - (4*nu2[i]*Sb[i][0][1]*math.sin(2*theta)) #Sigma Axial
                Stt[j] = Sb[i][0][0] + Sb[i][1][1] -(2*(Sb[i][0][0] - Sb[i][1][1])*math.cos(2*theta)) - (4*Sb[i][0][1]*math.sin(2*theta)) - deltaP -sigmaT #Sigma Hoop
                Ttz[j] = 2*((Sb[i][1][2]*math.cos(theta))-(Sb[i][0][2]*math.sin(theta)))#Tao Hoop
                Srr[j] = deltaP #Sigma Radial
                STMax[j] = 0.5*(Szz[j] + Stt[j] + (((Szz[j]-Stt[j])**2)+(4*(Ttz[j]**2)))**0.5)
                Stmin[j] = 0.5*(Szz[j] + Stt[j] - (((Szz[j]-Stt[j])**2)+(4*(Ttz[j]**2)))**0.5)
                ladempa[j] = mod_lad_cmw(SbFF[i][0][0],SbFF[i][1][1],SbFF[i][2][2],SbFF[i][0][1],SbFF[i][0][2],SbFF[i][1][2],j,phi[i],lal[i],psipp[i]/145.038)
                #print(ladempa[j])
            hoopmax[i] = np.nanmax(STMax)
            hoopmin[i] = np.nanmin(Stmin)
            lademax[i] = np.nanmax(ladempa)
            #print(lademax[i])
            lademin[i] = np.nanmin(ladempa)
            #ladempa = mod_lad_cmw(SbFF[i][0][0],SbFF[i][1][1],SbFF[i][2][2],SbFF[i][0][1],SbFF[i][0][2],SbFF[i][1][2],j,phi[i],lal[i],psipp[i]/145.038)
        except:
            Sb[i] = np.full((3,3),np.nan)
            SbFF[i] = np.full((3,3),np.nan)
            hoopmax[i] = np.nan
            hoopmin[i] = np.nan
            lademax[i] = np.nan
            lademin[i] = np.nan
    Sby = interpolate_nan(SbFF[:,1,1])
    Sbx = interpolate_nan(SbFF[:,0,0])
    Sbminmpa = np.minimum(Sby,Sbx)
    Sbmaxmpa = np.maximum(Sby,Sbx)
    Sbmingcc = ((Sbminmpa*145.038)/tvdf)/0.4335275040012
    Sbmaxgcc = ((Sbmaxmpa*145.038)/tvdf)/0.4335275040012
    """plt.plot(interpolate_nan(SbFF[:,0,0]),tvd, label='aligned sx')
    plt.plot(interpolate_nan(SbFF[:,1,1]),tvd, label='aligned sy')
    plt.plot(interpolate_nan(SbFF[:,2,2]),tvd, label='aligned sz')
    plt.plot(psifg/145.038,tvd, alpha=0.5, label='initial shm')
    plt.plot(sgHMpsi/145.038,tvd, alpha=0.5, label='initial sHM')
    plt.plot(obgpsi/145.038,tvd, alpha=0.5, label='initial sV')
    plt.legend()
    plt.show()
    plt.close()"""
    print("calculation complete")
    
    #ladempa = mod_lad_cmw(psifg/145.038,sgHMpsi/145.038,obgpsi/145.038,np.zeros(len(obgpsi)),np.zeros(len(obgpsi)),np.zeros(len(obgpsi)),offset-90,phi,lal,psipp/145.038)
    #ladempa = mod_lad_cmw(hoopmin,hoopmax,Sb[:,2,2],np.zeros(len(obgpsi)),np.zeros(len(obgpsi)),np.zeros(len(obgpsi)),offset-90,phi,lal,psipp/145.038)
    mogimpa = mogi(psifg/145.038,sgHMpsi/145.038,obgpsi/145.038)
    
    ladegcc = ((lademax*145.038)/tvdf)/0.4335275040012
    mogigcc = ((mogimpa*145.038)/tvdf)/0.4335275040012
    ladegcc = interpolate_nan(ladegcc)
    
    """plt.plot(psifg/145.038)
    plt.plot(ladempa)
    plt.plot(mogimpa)
    plt.show()
    plt.close()"""
    
    """print(gr)
    print(dalm)
    print(dtNormal)
    print(mudweight)
    print(fg.as_numpy())
    print(pp.as_numpy())
    print(obgcc.as_numpy())
    print(fgpsi.as_numpy())
    print(ssgHMpsi)
    print(obgpsi)
    print(hydropsi)
    print(pppsi.as_numpy())
    print(mudpsi)
    print(sgHMpsiL)
    print(sgHMpsiU)
    print(slal)
    print(shorsud)"""
    
    results = pd.DataFrame({
        'dalm': dalm,
        'dtNormal': dtNormal,
        'mudweight': mudweight*ureg.gcc,
        'fg': fg.as_numpy()*ureg.gcc,
        'pp': pp.as_numpy()*ureg.gcc,
        'sfg':ladegcc*ureg.gcc,
        'obgcc': obgcc.as_numpy()*ureg.gcc,
        'fgpsi': fgpsi.as_numpy()*ureg.psi,
        'ssgHMpsi': ssgHMpsi*ureg.psi,
        'obgpsi': obgpsi*ureg.psi,
        'hydropsi': hydropsi*ureg.psi,
        'pppsi': pppsi.as_numpy()*ureg.psi,
        'mudpsi': mudpsi*ureg.psi,
        'sgHMpsiL': sgHMpsiL*ureg.psi,
        'sgHMpsiU': sgHMpsiU*ureg.psi,
        'slal': slal*ureg.MPa,
        'shorsud': shorsud*ureg.MPa,
        'GR': gr,
        'GR_CUTOFF': grcut
    }, index=tvdm*ureg.m)
    
    def convert_units(data, pressure_unit, gradient_unit, strength_unit):
        converted_data = data.copy()
        
        # Define unit mappings
        unit_mappings = {
            'pressure': {'psi': ureg.psi, 'ksc': ureg.ksc, 'bar': ureg.bar, 'atm': ureg.atm, 'MPa': ureg.MPa},
            'gradient': {'gcc': ureg.gcc, 'sg': ureg.sg, 'ppg': ureg.ppg, 'psi/foot': ureg.psi/ureg.foot, 'ksc/m': ureg.ksc/ureg.m},
            'strength': {'MPa': ureg.MPa, 'psi': ureg.psi, 'ksc': ureg.ksc, 'bar': ureg.bar, 'atm': ureg.atm},
            'depth': {'m': ureg.m, 'f': ureg.foot, 'km': ureg.km, 'mile': ureg.mile, 'nm': ureg.nautical_mile, 'in': ureg.inch, 'cm': ureg.cm, 'fathom': ureg.fathom}
        }
        
        # Convert pressure columns
        pressure_columns = ['fgpsi', 'ssgHMpsi', 'obgpsi', 'hydropsi', 'pppsi', 'mudpsi', 'sgHMpsiL', 'sgHMpsiU']
        for col in pressure_columns:
            if col in converted_data.columns:
                converted_data[col] = (converted_data[col].values * ureg.psi).to(unit_mappings['pressure'][pressure_unit])
        
        # Convert gradient columns
        gradient_columns = ['mudweight', 'fg', 'pp', 'sfg', 'obgcc']
        for col in gradient_columns:
            if col in converted_data.columns:
                converted_data[col] = (converted_data[col].values * ureg.gcc).to(unit_mappings['gradient'][gradient_unit])
        
        # Convert strength columns
        strength_columns = ['slal', 'shorsud']
        for col in strength_columns:
            if col in converted_data.columns:
                converted_data[col] = (converted_data[col].values * ureg.MPa).to(unit_mappings['strength'][strength_unit])
        
        # Convert depth index
        converted_data.index = (converted_data.index.values * ureg.m).to(ul[unitchoice[0]]).magnitude
        
        return converted_data

    def convert_points_data(points_data, pressure_unit, gradient_unit, strength_unit):
        converted_points = {}
        
        for key, (x_vals, y_vals) in points_data.items():
            # Convert x values based on their type
            if key in ['frac_grad', 'flow_grad']:
                x_vals = (np.array(x_vals) * ureg.gcc).to(ureg(gradient_unit)).magnitude
            elif key in ['frac_psi', 'flow_psi']:
                x_vals = (np.array(x_vals) * ureg.psi).to(ureg(pressure_unit)).magnitude
            elif key == 'ucs':
                x_vals = (np.array(x_vals) * ureg.MPa).to(ureg(strength_unit)).magnitude
            
            # Convert depth values
            #y_vals = (np.array(y_vals) * ureg.m).to(ul[unitchoice[0]]).magnitude
            
            converted_points[key] = (x_vals, y_vals)
        
        return converted_points
    
    #unitchoice = [1,0,2,0,0] #pressure, strength, gradient, length, temperature
    try:
        with open(unitpath, 'r') as f:
            reader = csv.reader(f)
            unitchoice = next(reader)
            unitchoice = [int(x) for x in unitchoice]  # Convert strings to integers
    except:
        unitchoice = [0,0,0,0,0] #Depth, pressure,gradient, strength, temperature
    # Data preparation for plot_logs
    pressure_unit = up[unitchoice[1]]  # Get the selected pressure unit
    gradient_unit = ug[unitchoice[2]]  # Get the selected gradient unit
    print(gradient_unit)
    strength_unit = us[unitchoice[3]]  # Get the selected strength unit (using the same as pressure)
    depth_unit = ul[unitchoice[0]]  # Get the selected depth unit
    
    #maxchartpressure = round(maxchartpressure.to(pressure_unit).magnitude)# This is now computed inside the dataframes function
    #minpressure = round(minpressure.to(pressure_unit).magnitude)# This is now computed inside the dataframes function
    
    data = convert_units(results, pressure_unit, gradient_unit, strength_unit)
    """pd.DataFrame({
        'dalm': dalm,
        'dtNormal': dtNormal,
        'mudweight': mudweight,
        'fg': fg.as_numpy(),
        'pp': pp.as_numpy(),
        'sfg':ladegcc,
        'obgcc': obgcc.as_numpy(),
        'fgpsi': fgpsi.as_numpy(),
        'ssgHMpsi': ssgHMpsi,
        'obgpsi': obgpsi,
        'hydropsi': hydropsi,
        'pppsi': pppsi.as_numpy(),
        'mudpsi': mudpsi,
        'sgHMpsiL': sgHMpsiL,
        'sgHMpsiU': sgHMpsiU,
        'slal': slal,
        'shorsud': shorsud,
        'GR': gr,
        'GR_CUTOFF': grcut
    }, index=tvdm)"""
    #print(data)
    # Define styles for the new plotter function
    styles = read_styles_from_file(minpressure,maxchartpressure,pressure_unit,strength_unit,gradient_unit)
    
    print("max pressure is ",maxchartpressure)
        
    # Convert y values to tvd
    def convert_to_tvd(y_values):
        if unitchoice[0]==0:
            return [tvd[find_nearest_depth(md, y)[0]] for y in y_values]
        else:
            return [tvdf[find_nearest_depth(md, y)[0]] for y in y_values]

    # Convert data points to DataFrame
    def create_points_dataframe(points_data):
        points_df = {}
        for key, (x_vals, y_vals) in points_data.items():
            y_vals_tvd = convert_to_tvd(y_vals)
            points_df[key] = pd.Series(data=x_vals, index=y_vals_tvd)
        points_df = pd.DataFrame(points_df)
        
        # Handle duplicate indices
        points_df = points_df.groupby(points_df.index).first()
        
        # Drop the first row
        #points_df = points_df.iloc[1:]
        
        # Replace zero values with NaN
        points_df = points_df.replace(0, np.nan)
        
        # Ensure UCS column is present
        if 'ucs' not in points_df.columns:
            points_df['ucs'] = np.nan
        
        return points_df
    
    #casing_dia2 = [[-x, y] for x, y in casing_dia]
    #casing_dia = casing_dia + [[-x, y] for x, y in casing_dia]
    # Gather points data
    points_data = {
        'frac_grad': zip(*frac_grad_data),
        'flow_grad': zip(*flow_grad_data),
        'frac_psi': zip(*frac_psi_data),
        'flow_psi': zip(*flow_psi_data)
    }
    print("casing points",casing_dia)
    print("Points:",flow_grad_data)
    if UCSs is not None:
        # Swap the columns in the ucss array
        ucss = np.array([[depth, ucs] for ucs, depth in ucss])
        points_data['ucs'] = zip(*ucss)
    
    pointstyles = read_pstyles_from_file(minpressure, maxchartpressure, pressure_unit, strength_unit, gradient_unit)
    
    # Add CALIPER styles only if cald is not empty
    if np.any(~np.isnan(cald)) > 0 or len(casing_dia)>1:
        print("Track 5 added")
        styles.update({
            'CALIPER1': {"color": "brown", "linewidth": 0.5, "style": '-', "track": 5, "left": -15, "right": 15, "type": 'linear', "unit": "in"},
            'CALIPER3': {"color": "brown", "linewidth": 0.5, "style": '-', "track": 5, "left": -15, "right": 15, "type": 'linear', "unit": "in"}
        })
        data['CALIPER1'] = cald / 2
        data['CALIPER3'] = cald / (-2)
        pointstyles.update({
        'casingshoe': {'color': 'black', 'pointsize': 30, 'symbol': 1, 'track': 5, 'left': -15, 'right': 15, 'type': 'linear', 'unit': 'in', 'uptosurface':True},
        'casingshoe2': {'color': 'black', 'pointsize': 30, 'symbol': 0, 'track': 5, 'left': -15, 'right': 15, 'type': 'linear', 'unit': 'in', 'uptosurface':True}
        })
        casing_dia2 = [[-x, y] for x, y in casing_dia]
        points_data['casingshoe'] = zip(*casing_dia)
        points_data['casingshoe2'] = zip(*casing_dia2)
    
    converted_points = convert_points_data(points_data, pressure_unit, gradient_unit, strength_unit)
    print(converted_points)
    points_df = create_points_dataframe(converted_points)
    #points_df = create_points_dataframe(points_data)
    # Ensure the points DataFrame handles missing data gracefully
    points_df = points_df.apply(lambda col: col.dropna())
    print(points_df)
    
    # Plot using plot_logs
    dpif=300
    if dpif<100:
        dpif=100
    if dpif>900:
        dpif=900
    figname = wella.uwi if wella.uwi != "" and wella.uwi != None else wella.name
    fig, axes = plot_logs(data, styles, y_min=(float(tango)*ureg('metre').to(ul[unitchoice[0]])).magnitude, y_max=(float(zulu)*ureg('metre').to(ul[unitchoice[0]])).magnitude, plot_labels=False,figsize=(15, 10),points=points_df,pointstyles=pointstyles,dpi=dpif,output_dir = output_dir)
    fig.suptitle("Wellbore : "+figname,fontsize=14,y=0.9)
    plt.savefig(output_file,dpi=dpif)
    choptop(20*(dpif/100), 0, os.path.join(output_dir,"BottomLabel.png"))
    cutify2(output_file,os.path.join(output_dir,"BottomLabel.png"),output_file,89*(dpif/100),99*(dpif/100),0,0)
    #chopify(output_file,119*6,109*6,120*6,120*6)
    plt.close()
    return df3

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


def datasets_to_las(path, datasets, custom_units={}, **kwargs):
    """
    Write datasets to a LAS file on disk.

    Args:
        path (Str): Path to write LAS file to
        datasets (Dict['<name>': pd.DataFrame]): Dictionary maps a
            dataset name (e.g. 'Curves') or 'Header' to a pd.DataFrame.
        curve_units (Dict[str, str], optional): Dictionary mapping curve names to their units.
            If a curve's unit is not specified, it defaults to an empty string.
    Returns:
        Nothing, only writes in-memory object to disk as .las
    """
    from functools import reduce
    import warnings
    from datetime import datetime
    from io import StringIO
    from urllib import error, request

    import lasio
    import numpy as np
    import pandas as pd
    from lasio import HeaderItem, CurveItem, SectionItems
    from pandas._config.config import OptionError

    from welly.curve import Curve
    from welly import utils
    from welly.fields import curve_sections, other_sections, header_sections
    from welly.utils import get_columns_decimal_formatter, get_step_from_array
    from welly.fields import las_fields as LAS_FIELDS
    # ensure path is working on every dev set-up
    path = utils.to_filename(path)

    # instantiate new LASFile to parse data & header to
    las = laua.LASFile()

    # set header df as variable to later retrieve curve meta data from
    header = datasets['Header']
    
    extracted_units = {}
    if not header.empty:
        curve_header = header[header['section'] == 'Curves']
        for _, row in curve_header.iterrows():
            if row['unit']:  # Ensure there is a unit specified
                extracted_units[row['original_mnemonic']] = row['unit']

    # Combine extracted units with custom units, custom units take precedence
    all_units = {**extracted_units, **custom_units}
    
    column_fmt = {}
    for curve in las.curves:
        column_fmt[curve.mnemonic] = "%10.5f"
    
    # unpack datasets
    for dataset_name, df in datasets.items():

        # dataset is the header
        if dataset_name == 'Header':
            # parse header pd.DataFrame to LASFile
            for section_name in set(df.section.values):
                # get header section df
                df_section = df[df.section == section_name]

                if section_name == 'Curves':
                    # curves header items are handled in curve data loop
                    pass

                elif section_name == 'Version':
                    if len(df_section[df_section.original_mnemonic == 'VERS']) > 0:
                        las.version.VERS = df_section[df_section.original_mnemonic == 'VERS']['value'].values[0]
                    if len(df_section[df_section.original_mnemonic == 'WRAP']) > 0:
                        las.version.WRAP = df_section[df_section.original_mnemonic == 'WRAP']['value'].values[0]
                    if len(df_section[df_section.original_mnemonic == 'DLM']) > 0:
                        las.version.DLM = df_section[df_section.original_mnemonic == 'DLM']['value'].values[0]

                elif section_name == 'Well':
                    las.sections["Well"] = SectionItems(
                        [HeaderItem(r.original_mnemonic,
                                    r.unit,
                                    r.value,
                                    r.descr) for i, r in df_section.iterrows()])

                elif section_name == 'Parameter':
                    las.sections["Parameter"] = SectionItems(
                        [HeaderItem(r.original_mnemonic,
                                    r.unit,
                                    r.value,
                                    r.descr) for i, r in df_section.iterrows()])

                elif section_name == 'Other':
                    las.sections["Other"] = df_section['descr'].iloc[0]

                else:
                    m = f"LAS Section was not recognized: '{section_name}'"
                    warnings.warn(m, stacklevel=2)

        # dataset contains curve data
        if dataset_name in curve_sections:
            header_curves = header[header.section == dataset_name]
            for column_name in df.columns:
                curve_data = df[column_name]
                curve_unit = all_units.get(column_name, '')  # Use combined units
                # Assuming header information for each curve is not available
                las.append_curve(mnemonic=column_name,
                                 data=curve_data,
                                 unit=curve_unit,
                                 descr='',
                                 value='')


    # numeric null value representation from the header (e.g. # -9999)
    try:
        null_value = header[header.original_mnemonic == 'NULL'].value.iloc[0]
    except IndexError:
        null_value = -999.25
    las.null_value = null_value

    # las.write defaults to %.5 decimal points. We want to retain the
    # number of decimals. We first construct a column formatter based
    # on the max number of decimal points found in each curve.
    if 'column_fmt' not in kwargs:
        kwargs['column_fmt'] = column_fmt

    # write file to disk
    with open(path, mode='w') as f:
        las.write(f, **kwargs)

#def on_plotPPzhang_done(self,future):
    #result = future.result()  # Get the result from plotPPzhang
    # Update the GUI with the result
    # This might involve displaying the plot or showing a notification




def main():
    app = MyApp('WellMasterGeoMech', 'com.example.porepressurebuddy')
    return app

if __name__ == "__main__":
    app = MyApp("WellMasterGeoMech", "in.rocklab.porepressurebuddy")
    app.main_loop()

