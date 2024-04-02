import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga import Window

import lasio as laua
import welly
#from smoothass import readDevFromAsciiHeader, plotPPzhang
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
import time

user_home = os.path.expanduser("~/documents")
app_data = os.getenv("APPDATA")
output_dir = os.path.join(user_home, "pp_app_plots")
output_dir1 = os.path.join(user_home, "pp_app_data")
input_dir = os.path.join(user_home, "pp_app_models")
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

output_file = os.path.join(output_dir, "PlotFigure.png")
output_fileS = os.path.join(output_dir, "PlotStability.png")
output_fileSP = os.path.join(output_dir, "PlotPolygon.png")
output_fileVec = os.path.join(output_dir, "PlotVec.png")
output_fileBHI = os.path.join(output_dir, "PlotBHI.png")
output_file2 = os.path.join(output_dir1, "output.csv")
output_file3 = os.path.join(output_dir1, "output.las")
modelpath = os.path.join(input_dir, "model.csv")
aliaspath = os.path.join(input_dir, "alias.txt")
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

modelheader = "RhoA,AMC_exp,NCT_exp,dtML,dtMAT,EATON_fac,perm_cutoff,window,start,stop,w_den,re_sub,tec_fac,A_dep,SHM_azi,tilt,nu_shale,su_sst,nu_lst,dt_lst"
defaultmodel = "17,0.8,0.0008,250,60,0.35,0.35,21,2500,2900,1.025,1,0,3500,0,0,0.32,0.27,0.25,65"
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
    def startup(self):
        
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
                entry2 = toga.TextInput(style=Pack(padding_left=2, width=100))
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
            
            depth_label = toga.Label("Casing Shoe Depth (m)", style=Pack(padding_right=2,text_direction='rtl'))
            depth_entry = toga.TextInput(style=Pack(padding_left=5, width=100), value="0")
            row_box.add(depth_label)
            row_box.add(depth_entry)

            mud_weight_label = toga.Label("Max. Mud Weight", style=Pack(padding_right=2,text_direction='rtl'))
            mud_weight_entry = toga.TextInput(style=Pack(padding_left=5, width=100), value="1")
            row_box.add(mud_weight_label)
            row_box.add(mud_weight_entry)
            
            od_label = toga.Label("Casing OD (inches)", style=Pack(padding_right=2,text_direction='rtl'))
            od_entry = toga.TextInput(style=Pack(padding_left=5, width=100), value="0")
            row_box.add(od_label)
            row_box.add(od_entry)
            
            bitdia_label = toga.Label("Bit Dia (inches)", style=Pack(padding_right=2,text_direction='rtl'))
            bitdia_entry = toga.TextInput(style=Pack(padding_left=5, width=100), value="0")
            row_box.add(bitdia_label)
            row_box.add(bitdia_entry)
            
            iv_label = toga.Label("Casing volume (bbl/100ft)", style=Pack(padding_right=2,text_direction='rtl'))
            iv_entry = toga.TextInput(style=Pack(padding_left=5, width=100), value="0")
            row_box.add(iv_label)
            row_box.add(iv_entry)
            
            ppf_label = toga.Label("Casing Weight (ppf)", style=Pack(padding_right=5,text_direction='rtl'))
            ppf_entry = toga.TextInput(style=Pack(padding_left=2, width=100), value="0")
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
        self.page3 = toga.Box(style=Pack(direction=COLUMN, alignment='center'))
        
        # Create a container with ROW direction for plot and frac_grad_data
        plot_and_data_box = toga.Box(style=Pack(direction=ROW, flex=1))
        
        # Create a container for flow_grad_data rows
        self.flow_grad_data_box = toga.Box(style=Pack(direction=COLUMN,width=200))
        plot_and_data_box.add(self.flow_grad_data_box)

        # Add the buttons for Add row and Remove row
        flow_row_button_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
        self.add_flow_grad_button = toga.Button("Add PP Grad", on_press=lambda x: self.add_flow_grad_data_row(x, row_type='flow_grad'), style=Pack(width=100))
        self.add_flow_psi_button = toga.Button("Add PP BHP", on_press=lambda x: self.add_flow_grad_data_row(x, row_type='flow_psi'), style=Pack(width=100))
        flow_row_button_box.add(self.add_flow_grad_button)
        flow_row_button_box.add(self.add_flow_psi_button)
        self.flow_grad_data_box.add(flow_row_button_box)

        # Initialize the list of flow_grad_data rows
        self.flow_grad_data_rows = []
        self.add_flow_grad_data_row(None, row_type='flow_grad')
        self.add_flow_grad_data_row(None, row_type='flow_psi')
        
        self.bg3 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        plot_and_data_box.add(self.bg3)
        #spacer_box = toga.Box(style=Pack(flex=0.01))  # Add this spacer box
        #plot_and_data_box.add(spacer_box)
        
        # Move the frac_grad_data related components inside the plot_and_data_box
        self.frac_grad_data_box = toga.Box(style=Pack(direction=COLUMN,width=200))
        plot_and_data_box.add(self.frac_grad_data_box)
        
        
        # Add the buttons for Add row and Remove row
        row_button_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
        self.add_frac_grad_button = toga.Button("Add Frac Grad", on_press=lambda x: self.add_frac_grad_data_row(x, row_type='frac_grad'), style=Pack(width=100))
        self.add_frac_psi_button = toga.Button("Add Frac BHP", on_press=lambda x: self.add_frac_grad_data_row(x, row_type='frac_psi'), style=Pack(width=100))
        row_button_box.add(self.add_frac_grad_button)
        row_button_box.add(self.add_frac_psi_button)
        self.frac_grad_data_box.add(row_button_box)

        # Initialize the list of frac_grad_data rows
        self.frac_grad_data_rows = []
        self.add_frac_grad_data_row(None, row_type='frac_grad')
        self.add_frac_grad_data_row(None, row_type='frac_psi')

        self.page3.add(plot_and_data_box)
        
        progressbox = toga.Box(style=Pack(direction=ROW, alignment='center'))
        self.progress = toga.ProgressBar(max=None,style=Pack(alignment='center', width = 512, height=5, flex=1))
        progressbox.add(self.progress)
        self.progress.stop()
        self.page3.add(self.progress)
        
        # Define the labels and default values
        global model
        entries_info = [
            {'label': 'RHOA (ppg)', 'default_value': str(model[0])},
            {'label': 'OBG Exp', 'default_value': str(model[1])},
            {'label': 'NCT Exp', 'default_value': str(model[2])},
            {'label': 'DTml (us/ft)', 'default_value': str(model[3])},
            {'label': 'DTmat (us/ft)', 'default_value': str(model[4])},
            {'label': "Eaton's Nu", 'default_value': str(model[5])},
            {'label': 'ShaleFlag Cutoff', 'default_value': str(model[6])},
            {'label': 'Window', 'default_value': str(model[7])},
            {'label': 'Start', 'default_value': str(model[8])},
            {'label': 'Stop', 'default_value': str(model[9])},
            {'label': 'WaterDensity', 'default_value': str(model[10])},
            {'label': 'PP Gr. L.Limit', 'default_value': str(model[11])},
            {'label': 'TectonicFactor', 'default_value': str(model[12])},
            {'label': 'Analysis TVD', 'default_value': "0"},
            {'label': 'Fast Shear Azimuth', 'default_value': "0"},
            {'label': 'Stress Tensor Tilt Beta', 'default_value': "0"},
            {'label': 'Stress Tensor Tilt Gamma', 'default_value': "0"}
            
        ]
        
        # Create a list to store the textboxes
        self.textboxes = []
        # Add 6 numeric entry boxes with their respective labels
        for i in range(2):
            entry_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
            for j in range(8):
                entry_info = entries_info[8 * i + j]
                label = toga.Label(entry_info['label'], style=Pack(alignment='center', width = 100, flex=1,text_direction='rtl'))  # Adjust flex as needed
                entry = toga.TextInput(style=Pack(width=50, flex=1))  # Adjust flex as needed
                entry.value = entry_info['default_value']
                entry_box.add(label)
                entry_box.add(entry)
                self.textboxes.append(entry)
            self.page3.add(entry_box)
        
        button_box = toga.Box(style=Pack(direction=ROW, alignment='center', flex=0))
        self.page3_btn1 = toga.Button("Recalculate", on_press=self.get_textbox_values, style=Pack(padding=1,width = 100))
        button_box.add(self.page3_btn1)
        
        self.page3_btn2 = toga.Button("Export Plot", on_press=self.save_plot, style=Pack(padding=1,width = 100))
        button_box.add(self.page3_btn2)
        
        self.page3_btn3 = toga.Button("Export Las", on_press=self.save_las, style=Pack(padding=1,width = 100))
        button_box.add(self.page3_btn3)
        
        self.page3_btn5 = toga.Button("Stability Plot", on_press=self.show_page4, style=Pack(padding=1,width = 100),enabled=False)
        button_box.add(self.page3_btn5)
        
        self.page3_btn4 = toga.Button("Back", on_press=self.show_page2, style=Pack(padding=1,width = 100))
        button_box.add(self.page3_btn4)
        
        
        
        self.page3.add(button_box)
        
        
        #Page4
        
        self.page4 = toga.Box(style=Pack(direction=COLUMN, alignment='center'))
        self.dbox = toga.Box(style=Pack(direction=ROW, alignment='center',flex=1))
        self.dbox2 = toga.Box(style=Pack(direction=ROW, alignment='center',flex=1))
        self.bg5 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        self.dbox.add(self.bg5)

        self.bg4 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        self.dbox.add(self.bg4)
        
        self.bg6 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        self.dbox2.add(self.bg6)

        self.bg7 = BackgroundImageView("BG2.png", style=Pack(flex = 1))
        self.dbox2.add(self.bg7)
        
        button_box4 = toga.Box(style=Pack(direction=ROW, alignment='center', flex=0))
        
        self.page4_btn1 = toga.Button("Start Over", on_press=self.show_page1, style=Pack(padding=1))
        button_box4.add(self.page4_btn1)
        
        self.page4_btn2 = toga.Button("Export Plot", on_press=self.save_las, style=Pack(padding=1))
        button_box4.add(self.page4_btn2)
        
        self.page4_btn3 = toga.Button("Back", on_press=self.show_page3, style=Pack(padding=1))
        button_box4.add(self.page4_btn3)
        
        self.page4.add(self.dbox)
        self.page4.add(self.dbox2)
        self.page4.add(button_box4)
        
        self.main_window = toga.MainWindow(title=self.formal_name,size=[1240,620])
        self.main_window.content = self.page1
        self.main_window.show()
        
    
    def add_frac_grad_data_row(self, widget, row_type='frac_grad'):
        depth_label = toga.Label("MD", style=Pack(text_align="center", width=100, padding_top=5))
        
        if row_type == 'frac_grad':
            second_label = toga.Label("Frac Grad gcc", style=Pack(text_align="center", width=100, padding_top=5))
        elif row_type == 'frac_psi':
            second_label = toga.Label("Frac BHP psi", style=Pack(text_align="center", width=100, padding_top=5))
        else:
            raise ValueError("Invalid row type")

        depth_input = toga.TextInput(style=Pack(width=100), value="0")
        second_input = toga.TextInput(style=Pack(width=100), value="0")

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
        depth_label = toga.Label("MD", style=Pack(text_align="center", width=100, padding_top=5))
        
        if row_type == 'flow_grad':
            second_label = toga.Label("PP Grad gcc", style=Pack(text_align="center", width=100, padding_top=5))
        elif row_type == 'flow_psi':
            second_label = toga.Label("PP BHP psi", style=Pack(text_align="center", width=100, padding_top=5))
        else:
            raise ValueError("Invalid row type")

        depth_input = toga.TextInput(style=Pack(width=100), value="0")
        second_input = toga.TextInput(style=Pack(width=100), value="0")

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
        global devpath,laspath,h1
        devpath = None
        laspath = None
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
                frac_grad_values.append([second_value,depth])
            elif row_type == 'frac_psi':
                frac_psi_values.append([second_value,depth])

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
                flow_grad_values.append([second_value, depth])
            elif row_type == 'flow_psi':
                flow_psi_values.append([second_value, depth])

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
            ppf_entry = row_box.children[11] #access the casing volume TextInput Widget

            try:
                depth = float(depth_entry.value)
                mw = float(mw_entry.value)
                od = float(od_entry.value)
                bd = float(bd_entry.value)
                iv = float(iv_entry.value)
                ppf = float(ppf_entry.value)
            except ValueError:
                print("Invalid input. Skipping this row.")
                continue

            depth_mw_values.append([mw,depth,bd,od,iv])

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
            laspath_dialog = await self.main_window.open_file_dialog(title="Select a las file", multiselect=False)
            if laspath_dialog:  # Check if the user selected a file and didn't cancel the dialog
                laspath = laspath_dialog  # This is where you access the selected path
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
        wella.location.ekb = tv[0]
        wella.location.kb = tv[0]
        wella.location.egl = tv[1]
        wella.location.gl = tv[1]
        #wella.location.tdl = tv[2]
        #wella.location.td = tv[2]
        wella.location.latitude = tv[3]
        wella.location.longitude = tv[4]
        wella.header.bht = tv[5]
        wella.header.rm = tv[6]
        wella.header.rmf = tv[7]
        attrib=tv
        print('Attributes set according to following:')
        print(attrib)
        print(wella.location.egl)
        #wella.unify_basis(keys=None, alias=None, basis=md)
    
    def plotppwrapper(self,*args, **kwargs):
        print("thread spawn")
        try:
            result = plotPPzhang(*args, **kwargs)
        except Exception as e:
            print(f"Error in thread: {e}")
        self.loop.call_soon_threadsafe(self.onplotfinish)
        print("Thread despawn")
        return
    
    def onplotfinish(self):
        self.bg3.image = toga.Image(output_file)
        self.bg3.refresh()
        self.page3_btn1.enabled = True
        self.page3_btn2.enabled = True
        self.page3_btn3.enabled = True
        self.page3_btn4.enabled = True
        self.progress.stop()
        if float(model[13])>0:
        
            self.page3_btn5.enabled = True
            self.bg4.image = toga.Image(output_fileS)
            self.bg5.image = toga.Image(output_fileSP)
            self.bg6.image = toga.Image(output_fileVec)
            self.bg7.image = toga.Image(output_fileBHI)
            self.bg4.refresh()
            self.bg5.refresh()
            self.bg6.refresh()
            self.bg7.refresh()
            #self.show_page4(widget)
        else:
            self.page3_btn5.enabled = False
        #self.progress.text = "Status: Done! Program Ready"
        print("Wrapper done")
        return
    
    def start_plotPPzhang_thread(self,*args, **kwargs):
    # Create a Thread to run plotPPzhang in the background
        #self.loop = asyncio.get_event_loop()
        thread = threading.Thread(target=self.plotppwrapper, args=args, kwargs=kwargs)#, on_close=self.onplotfinish)
        thread.start()
        #time.sleep(1)
        return
    
    async def get_textbox_values(self,widget):
        global wella
        global attrib
        global model
        
        self.progress.text = "Status: Calculating, Standby"
        #yield 0.1
        self.getwelldev()
        data = pd.read_csv(modelpath,index_col=False)
        data_into_list = data.values.tolist()
        print(data_into_list)
        model = data_into_list[0]
        tail = model[16:20]
        #tail2 = str(model[17])
        #tail3 = str(model[18])
        #tail4 = str(model[19])
        tv = [textbox.value for textbox in self.textboxes]
        self.bg3.image = toga.Image('BG1.png')
        self.bg4.image = toga.Image('BG1.png')
        self.bg5.image = toga.Image('BG1.png')
        self.bg6.image = toga.Image('BG1.png')
        self.bg7.image = toga.Image('BG1.png')
        #self.bg3.refresh()
        #self.bg4.refresh()
        #self.bg5.refresh()
        model = tv
        model = model + tail
        print(model)
        #model.append(tail2)
        #model.append(tail3)
        #model.append(tail4)
        file = open(modelpath,'w')
        file.write(modelheader+'\n')
        for item in model:
            file.write(str(item)+",")
        file.close()
        print("Great Success!! :D")
        image_path = 'PlotFigure.png'
        
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
        #executor = concurrent.futures.ProcessPoolExecutor()
        self.loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            ih = await self.loop.run_in_executor(pool,self.start_plotPPzhang_thread,wella, float(model[0]), float(model[2]), float(model[1]), float(model[5]), float(model[6]), int(float(model[7])), float(model[8]), float(model[9]), float(model[3]), float(model[4]),float(model[10]),float(model[11]),float(model[12]),float(model[13]),float(model[14]),float(model[15]))
        #self.onplotfinish()
        #self.progress.text = "Status: Done! Program Ready"
        #future.add_done_callback(lambda f: on_plotPPzhang_done(self, f))
        #ih = plotPPzhang(wella,self, float(model[0]), float(model[2]), float(model[1]), float(model[5]), float(model[6]), int(float(model[7])), float(model[8]), float(model[9]), float(model[3]), float(model[4]),float(model[10]),float(model[11]),float(model[12]),float(model[13]),float(model[14]),float(model[15]))
                
        
        
    
    
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
            'neutron': ['none', 'CNCF', 'NPHI', 'NEU']
        }
        # Convert aliases dictionary to JSON string and write to the file
        with open(file_path, 'w') as file:
            json.dump(aliases, file, indent=4)
        return aliases


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


def plotPPzhang(well,rhoappg = 16.33, lamb=0.0008, a = 0.630, nu = 0.4, sfs = 1.0, window = 1, zulu=0, tango=2000, dtml = 210, dtmt = 60, water = 1.0, underbalancereject = model[11] ,b = float(model[12]),doi = 0,offset = float(model[14]), tilt = 0, lala = -1.0, lalb = 1.0, lalm = 5, lale = 0.5, lall = 5, horsuda = 0.77, horsude = 2.93):
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
    
    header = well._get_curve_mnemonics()
    print(header)
    alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
    alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
    alias['ssonic'] = [elem for elem in header if elem in set(alias['ssonic'])]
    alias['resdeep'] = [elem for elem in header if elem in set(alias['resdeep'])]
    alias['resshal'] = [elem for elem in header if elem in set(alias['resshal'])]
    alias['density'] = [elem for elem in header if elem in set(alias['density'])]
    alias['neutron'] = [elem for elem in header if elem in set(alias['neutron'])]
    
    global mwvalues
    global flowgradvals
    global fracgradvals
    global flowpsivals
    global fracpsivals
    
    detail = mwvalues
    print(detail)
    i = 0
    mud_weight = []
    while i<len(detail):
        mud_weight.append([detail[i][0],detail[i][1]])
        i+=1    
    print(mud_weight)
    first = [mud_weight[0][0],0]
    last = [mud_weight[-1][0],final_depth]
    frac_grad_data = fracgradvals
    flow_grad_data = flowgradvals
    frac_psi_data = fracpsivals
    flow_psi_data = flowpsivals
    
    mud_weight.insert(0,first)
    mud_weight.append(last)
    print("MudWeights: ",mud_weight)
    
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
        gr = well.data[alias['gr'][0]].values
    except:
        zden2 = np.full(len(md),np.nan)
        gr = np.full(len(md),np.nan)
    
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
    global attrib
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
        ttvdlist = np.transpose(formlist)[4]
        ftvdlist = np.transpose(formlist)[0]
        ttvdlist = np.append(0,ttvdlist)
        fttvdlist = np.append(0,ftvdlist)
        difftvd = np.zeros(len(fttvdlist))
        hydrldiff = np.zeros(len(fttvdlist))
        htvdlist = np.zeros(len(fttvdlist))
        
        i=0
        while i<len(fttvdlist):
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
        ttvdlist=ttvdlist.astype(float)
        htvdlist=htvdlist.astype(float)
        print("Differential TVD list:")
        print([difftvd,fttvdlist,ttvdlist])
    

    
    j=0
    k=0
    m=0
    n=0
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
            hydrotvd[i] = np.interp(tvd[i],fttvdlist,htvdlist)

        i+=1
    
    #check plot
    """plt.plot(lithotype,md)
    plt.plot(nu3,md)
    plt.plot(nu2,md)
    plt.plot(mu2,md)
    plt.plot(ucs2,md)
    plt.plot(ilog,md)
    plt.show()
    plt.clf()"""
    #plt.plot(hydrotvd,tvd)
    #plt.show()
    #plt.clf()
    
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
                    ObgTppg[i] =((maxwaterppg + ((np.mean(rhoppg[0:i]))*(tvdbglf[i])))/tvdmslf[i])
                    integrho[i] = integrho[i-1]+(rhogcc[i]*9806.65*(tvdbgl[i]-tvdbgl[i-1])) #in pascals
                    integrhopsift[i] = (integrho[i]*0.000145038)/tvdf[i]
            else:
                if(tvdmsl[i]>0):
                    integrho[i] = integrho[i-1]+(water*9806.65*(tvdbgl[i]-tvdbgl[i-1])) #in pascals
                    integrhopsift[i] = (integrho[i]*0.000145038)/tvdf[i]
                    ObgTppg[i] =(8.34540426515252*water)
        else:
            if (tvdbgl[i]>0):
                integrho[i] = integrho[i-1]+(rhogcc[i]*9806.65*(tvdbgl[i]-tvdbgl[i-1])) #in pascals
                integrhopsift[i] = (integrho[i]*0.000145038)/tvdf[i]
                ObgTppg[i] =((np.mean(rhoppg[0:i]))*(tvdbglf[i]))/tvdf[i] #Curved Top Obg Gradient
                #ObgTppg[i] =rhoppg[i] #Flat Top Obg Gradient
        i+=1
    
    ObgTppg = integrhopsift*19.25
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
        hydropsi = hydroppf[:]*tvdmslf[:]
        obgpsi= integrho*0.000145038
        #obgpsi = np.array([np.mean(ObgTppf[0:i]) * tvdmslf[i-1] for i in range(1, len(ObgTppf) + 1)])
    else:
        hydropsi = hydroppf[:]*tvdbglf[:]
        #obgpsi = np.array([np.mean(ObgTppf[0:i]) * tvdbglf[i-1] for i in range(1, len(ObgTppf) + 1)])
        obgpsi= integrho*0.000145038
    mudpsi = mudppf[:]*tvdf[:]
    i = 0
    ppgZhang = np.zeros(len(tvdf))
    gccZhang = np.zeros(len(tvdf))
    psipp = np.zeros(len(tvdf))
    psiftpp = np.zeros(len(tvdf))
    horsud = np.zeros(len(tvdf))
    lal = np.zeros(len(tvdf))
    ym = np.zeros(len(tvdf))
    sm = np.zeros(len(tvdf))
    lal3 = np.zeros(len(tvdf))
    phi = np.zeros(len(tvdf))
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
    
    while i<(len(ObgTppg)-1):
        if glwd>=0: #Onshore Cases
            if tvdbgl[i]>0:
                if shaleflag[i]<0.5: #Shale PorePressure
                    gccZhang[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    #gccZhang[i] = getGccZhang(ObgTgcc[i],pn,mudline,matrick,dalm[i],ct,tvdbgl[i])
                else:
                    gccZhang[i] = np.nan #Hydraulic Pore Pressure
                
                if lithotype[i]>1.5: #Carbonate PorePressure
                    dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                    gccZhang[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    #gccZhang[i] = getGccZhang(ObgTgcc[i],pn,mudline,matrick,dalm[i],ct,tvdbgl[i])
                
                if gccZhang[i]<underbalancereject: #underbalance reject
                    gccZhang[i]=underbalancereject
                
                ppgZhang[i] = gccZhang[i]*8.33
                dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                lal3[i] = lall*(304.8/(dalm[i]-1))
                lal[i] = lalm*(vp[i]+lala)/(vp[i]**lale)
                horsud[i] = horsuda*(vp[i]**horsude)
                if np.isnan(ucs2[i]) or ucs2[i]==0:
                    ucs2[i] = horsud[i]
                phi[i] = np.arcsin(1-(2*nu2[i]))
                H[i] = (4*(np.tan(phi[i])**2))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i]))))
                K[i] = (4*lal[i]*(np.tan(phi[i])))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i])))) 
                ym[i] = 0.076*(vp[i]**3.73)
                sm[i] = 0.03*(vp[i]**3.30)
                psiftpp[i] = 0.4335275040012*gccZhang[i]
                psipp[i] = psiftpp[i]*tvdf[i]
                #if psipp[i]<hydropsi[i]:
                #   psipp[i] = hydropsi[i]
        else: #Offshore Cases
            if tvdbgl[i]>0:
                if shaleflag[i]<0.5:#Shale Pore Pressure
                    gccZhang[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    #gccZhang[i] = getGccZhang(ObgTgcc[i],pn,mudline,matrick,dalm[i],ct,tvdbgl[i])
                else:
                    gccZhang[i] = np.nan #Hydraulic Pore Pressure
                
                if lithotype[i]>1.5: #Carbonate PorePressure
                    dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                    gccZhang[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    #gccZhang[i] = getGccZhang(ObgTgcc[i],pn,mudline,matrick,dalm[i],ct,tvdbgl[i])
                
                if gccZhang[i]<underbalancereject: #underbalance reject
                    gccZhang[i]=underbalancereject
                
                ppgZhang[i] = gccZhang[i]*8.33
                dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                lal3[i] = lall*(304.8/(dalm[i]-1))
                lal[i] = lalm*(vp[i]+lala)/(vp[i]**lale)
                horsud[i] = horsuda*(vp[i]**horsude)
                if np.isnan(ucs2[i]) or ucs2[i]==0:
                    ucs2[i] = horsud[i]
                phi[i] = np.arcsin(1-(2*nu2[i]))
                H[i] = (4*(np.tan(phi[i])**2))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i]))))
                K[i] = (4*lal[i]*(np.tan(phi[i])))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i])))) 
                ym[i] = 0.076*(vp[i]**3.73)
                sm[i] = 0.03*(vp[i]**3.30)
                psiftpp[i] = 0.4335275040012*gccZhang[i]
                psipp[i] = psiftpp[i]*tvdf[i]
        i+=1
    #gccZhang[0] = np.nan
    gccZhang[-1] = hydrostatic[-1]
    gccZhang[np.isnan(gccZhang)] = water
    ppgZhang[np.isnan(ppgZhang)] = water*8.33
    psiftpp = interpolate_nan(psiftpp)
    psipp = interpolate_nan(psipp)
    print("GCCZhang: ",gccZhang)
    psiftpp = 0.4335275040012*gccZhang
    #psipp = psiftpp[:]*tvdf[:]
    #ObgTgcc = np.array(ObgTgcc)
    #obgpsift = 0.4335275040012*ObgTgcc
    
    #plt.plot(ucs2,md)
    #plt.plot(dtNormal)
    #plt.plot(dalm)
    ###plt.show()
    #plt.clf()
    
    #Eatons/Daines
    
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
                fgppg[i] = (nu2[i]/(1-nu2[i]))*(ObgTppg[i]-ppgZhang[i])+ppgZhang[i] +(b*ObgTppg[i])
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
    
    
    #from DrawSP import getSHMax_optimized
    from DrawSP import getSP
    from DrawSP import drawSP
    
    i=0
    sgHMpsi = np.zeros(len(tvd))
    sgHMpsiL = np.zeros(len(tvd))
    sgHMpsiU = np.zeros(len(tvd))
    psisfl = np.zeros(len(tvd))
    while i<len(tvd)-1:
        result = getSP(obgpsi[i]/145.038,psipp[i]/145.038,mudpsi[i]/145.038,psifg[i]/145.038,ucs2[i],phi[i],ilog[i],mu2[i])
        sgHMpsi[i] = (result[2])*145.038
        sgHMpsiL[i] = (result[0])*145.038
        sgHMpsiU[i] = (result[1])*145.038
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
        drawSP(output_fileSP,sigmaVmpa,ppmpa,bhpmpa,sigmahminmpa,ucsmpa,phi[doiX],ilog_flag,mu2[doiX])
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
        gamma=0
        from BoreStab import getRota
        Rmat = getRota(alpha,beta,gamma)
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
        osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alpha,beta,gamma)
        sten = getStens(osx,osy,osz,alpha,beta,gamma)
        sn,se,sd = np.linalg.eigh(sten)[0]
        on,oe,od = np.linalg.eigh(sten)[1]
        savevec(on,oe,od,2,sn,se,sd,output_fileVec)
        #drawStab(sigmas[0],sigmas[1],sigmas[2],sigmas[3],alpha,beta,gamma)
        draw(output_fileS,tvd[doiX],osx,osy,osz,sigmas[3],sigmas[4],ucsmpa,alpha,beta,gamma,0,nu2[doiX],incdoi,azmdoi)
        
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
    
    from BoreStab import getHoop
    from plotangle import plotfracs
    def drawBHimage(doi):
        doiactual = find_nearest_depth(tvdm,doi-25)
        doiS = doiactual[0]
        doiactual2 = find_nearest_depth(tvdm,doi+25)
        doiF = doiactual2[0]
        frac = np.zeros([doiF-doiS,361])
        crush = np.zeros([doiF-doiS,361])
        data=np.zeros([doiF-doiS,4])
        data2=np.zeros([doiF-doiS,3])
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
            osx,osy,osz = get_optimal(sigmas[0],sigmas[1],sigmas[2],alpha,beta,gamma)
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
            cr,fr,minazi,maxazi,minangle,maxangle = getHoop(incdoi,azmdoi,sigmas[0],sigmas[1],sigmas[2],deltaP,ppmpa,ucsmpa,alpha,beta,gamma,nu2[i])
            crush[j] = cr
            frac[j] = fr
            if np.max(frac[j])>0:
                data[j] = [tvd[i],minazi,minangle,maxangle]
                #data2[j+1] = [tvd[i+1],round((minazi+180)%360),minangle+180]
            i+=1
            j+=1
        from plotangle import plotfracs
        plotfracs(data)
        plt.imshow(frac,cmap='Reds',alpha=0.5,extent=[0,360,tvd[doiF],tvd[doiS]],aspect=10)
        plt.imshow(crush,cmap='Blues',alpha=0.5,extent=[0,360,tvd[doiF],tvd[doiS]],aspect=10)
        plt.title("Synthetic Borehole Image")
        plt.savefig(output_fileBHI,dpi=1200)
        plt.clf()
        print(crush)
        print(frac)
    
    if doi>0:
        drawBHimage(doi)
    
    """
    #Preview Plot
    graph, (plt1, plt2,plt3,plt4,plt5,plt6,plt7) = plt.subplots(1, 7,sharey=True)
    plt5.plot(slal,tvd,label='lal-direct-c0')
    plt5.plot(slal2,tvd,label='horsud E')
    plt5.plot(slal3,tvd,label='horsud G')
    plt5.plot(shorsud,tvd, label='horsud UCS')
    plt5.invert_yaxis()
    plt5.legend()
    #plt5.set_xlim([0,20])
    
    plt1.plot(gr,tvd,color='green',label='GR')
    plt1.legend()
    plt1.set_xlim([0,150])
    
    plt2.plot(dalm,tvd,label='DT')
    plt2.plot(dtNormal,tvd,label='Normal DT (Zhang)')
    plt2.legend()
    plt2.set_xlim([300,50])

    plt3.plot(mudweight,tvd,color='brown',label='Mud Gradient')
    plt3.plot(fg,tvd,color='blue',label='Fracture Gradient (Daines)')
    #plt3.plot(fg2,tvd,color='aqua',label='Fracture Gradient (Zoback)')
    plt3.plot(pp,tvd,color='red',label='Pore Pressure Gradient (Zhang)')
    plt3.plot(obgcc,tvd,color='lime',label='Overburden (Amoco)')
    plt3.legend()
    plt3.set_xlim([0,3])

    plt4.plot(fgpsi,tvd,color='blue',label='Sh min')
    plt4.plot(ssgHMpsi,tvd,color='pink',label='SH Max')
    plt4.plot(obgpsi,tvd,color='green',label='Sigma V')
    plt4.plot(hydropsi,tvd,color='aqua',label='Hydrostatic')
    plt4.plot(pppsi,tvd,color='red',label='Pore Pressure')
    plt4.plot(mudpsi,tvd,color='red',label='Bottom Hole Pressure')
    plt4.legend()
    #plt4.set_xlim([0,5000])
    
    litho.plot_2d(plt6,cmap='seismic')
    coal.plot_2d(plt7,cmap='binary')
    #plt6.plot(fgpsi,tvd,color='blue',label='Sh min')
    #plt6.plot(obgpsi,tvd,color='green',label='Sigma V')
    #plt2.plot(dalm,tvd,label='DT')
    #plt2.legend()
    #plt2.set_xlim([0,150])
    
    plt.show()
    plt.clf()
    
    
    
    
    
    #well.data['Gcal'] =  gcal
    #well.data['Pcal'] =  pcal

    #well.data['DESPGR'] = gr.despike(z=1)
    #well.data['DIFFGR'] = gr - well.data['DESPGR']
    well.data['TVDM'] =  TVDM
    well.data['TVDBGL'] =  TVDBGL
    well.data['TVDF'] =  TVDF
    well.data['TVDMSL'] =  TVDMSL
    
    #plt = well.plot(tracks=['GR', 'DESPGR', 'DIFFGR'])
    ###plt.show()
    plt.clf()
    """
    
    """plt5.plot(slal,tvd,label='lal-direct-c0')
    plt5.plot(slal2,tvd,label='horsud E')
    plt5.plot(slal3,tvd,label='horsud G')
    plt5.plot(shorsud,tvd, label='horsud UCS')
    
    plt5.legend()
    plt5.set_xlim([0,20])
    
    
    plt1.plot(gr,tvd,color='green',label='GR')
    plt1.legend()
    plt1.set_xlim([0,150])"""
    
    from matplotlib import ticker
    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
    
    tango = min(tango,finaldepth)
    if zulu>finaldepth or zulu>tango:
        zulu=0
    #minindex = find_nearest_depth(tvdm,zulu)[0]
    #maxindex = find_nearest_depth(tvdm,tango)[0]
    #Presentation Plot
    plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labelbottom'] = False
    
    graph, splt = plt.subplots(1, 6,sharey=True)
    graph.suptitle(well.name,fontsize=18)
    #splt[0].invert_yaxis()
    splt[2].set_ylim([tango,zulu])
    splt[2].plot(dalm,tvd,label='DT')
    splt[2].plot(dtNormal,tvd,label='Normal DT (Zhang)')
    splt[2].legend(bbox_to_anchor=(0.5, 0),fontsize = "6",loc='upper center')
    splt[2].set_xlim([300,50])
    splt[2].yaxis.set_major_locator(MultipleLocator(100))
    splt[2].yaxis.set_minor_locator(MultipleLocator(10))
    #splt[0].tick_params(axis='both', which='minor', length = 0)
    #splt[0].set_yticks([zulu,tango,10])
    #splt[0].set_yticks([zulu,tango,1],minor=True)
    splt[2].grid(which='both')
    splt[2].grid(which='minor', color = 'teal', alpha = 0.025)
    splt[2].grid(which='major', color = 'teal', alpha = 0.05)
    #N = 11
    #ymin, ymax = splt[0].get_ylim()
    #splt[0].set_yticks(np.round(np.linspace(ymin, ymax, N)))
    
    #N1 = 6
    splt[2].xaxis.set_major_locator(MultipleLocator(100))
    splt[2].xaxis.set_minor_locator(MultipleLocator(10))
    #xmin, xmax = splt[0].get_xlim()
    #splt[0].set_xticks(np.round(np.linspace(xmin, xmax, N1)))
    splt[2].title.set_text("Sonic (us/ft)")

    
    splt[3].plot(mudweight,tvd,color='brown',label='Mud Gradient')
    splt[3].plot(fg,tvd,color='blue',label='Shmin Gradient (Daines)')
    #splt[1].plot(fg2,tvd,color='aqua',label='Fracture Gradient (Zoback)')
    splt[3].plot(pp,tvd,color='red',label='Pore Pressure Gradient (Zhang)')
    splt[3].plot(obgcc,tvd,color='lime',label='Overburden (Amoco)')
    splt[3].legend(bbox_to_anchor=(0.5, 0),fontsize = "6",loc='upper center')
    splt[3].set_xlim([0,3])
    splt[3].grid(which='both')
    splt[3].grid(which='minor', color = 'teal', alpha = 0.025)
    splt[3].grid(which='major', color = 'teal', alpha = 0.05)
    splt[3].xaxis.set_major_locator(MultipleLocator(1))
    splt[3].xaxis.set_minor_locator(MultipleLocator(0.1))
    N2 = 4
    xmin, xmax = splt[3].get_xlim()
    splt[3].set_xticks(np.round(np.linspace(xmin, xmax, N2),1))
    splt[3].title.set_text("Gradients (g/cc)")

    splt[4].plot(fgpsi,tvd,color='blue',label='Sh min')
    splt[4].plot(ssgHMpsi,tvd,color='pink',label='SH MAX ML')
    splt[4].plot(obgpsi,tvd,color='green',label='Sigma V')
    splt[4].plot(hydropsi,tvd,color='aqua',label='Hydrostatic')
    splt[4].plot(pppsi,tvd,color='red',label='Pore Pressure')
    splt[4].plot(mudpsi,tvd,color='brown',label='BHP')
    splt[4].plot(sgHMpsiL,tvd,color='lime',lw=0.25,label='SH MAX L')
    splt[4].plot(sgHMpsiU,tvd,color='orange',lw=0.25,label='SH MAX U')
    splt[4].grid(which='both')
    splt[4].grid(which='minor', color = 'teal', alpha = 0.025)
    splt[4].grid(which='major', color = 'teal', alpha = 0.05)
    splt[4].set_xlim(left=0)
    splt[4].legend(bbox_to_anchor=(0.5, 0),fontsize = "6",loc='upper center')
    splt[4].xaxis.set_major_locator(MultipleLocator(5000))
    splt[4].xaxis.set_minor_locator(MultipleLocator(500))
    xlabels = ['{:,.0f}'.format(x) + 'K' for x in splt[4].get_xticks()/1000]
    splt[4].set_xticklabels(xlabels)
    #N3 = 4
    xmin, xmax = splt[4].get_xlim()
    #splt[2].set_xticks(np.round(np.linspace(xmin, xmax, N3)))
    splt[4].title.set_text("Pressures (psi)")

    
    splt[5].set_ylim([tango,zulu])
    splt[5].plot(slal2,tvd,label='UCS (Lal)')
    splt[5].plot(shorsud,tvd,label='UCS (Horsud)')
    splt[5].legend(bbox_to_anchor=(0.5, 0),fontsize = "6",loc='upper center')
    splt[5].set_xlim([0,100])
    splt[5].grid(which='both')
    splt[5].grid(which='minor', color = 'teal', alpha = 0.025)
    splt[5].grid(which='major', color = 'teal', alpha = 0.05)
    splt[5].xaxis.set_major_locator(MultipleLocator(20))
    splt[5].xaxis.set_minor_locator(MultipleLocator(2))
    #N4 = 6
    #xmin, xmax = splt[3].get_xlim()
    #splt[3].set_xticks(np.round(np.linspace(xmin, xmax, N4)))
    splt[5].title.set_text("Strengths (MPa)")
    
    from CasingPlotter import plotCasing
    shoes = np.transpose(mud_weight)[1]
    tvdshoes = [tvd[find_nearest_depth(md,shoe)[0]] for shoe in shoes]
    plotCasing(tvdshoes,splt[0],(tango-zulu)/80)
    splt[0].set_ylim([tango,zulu])
    #splt[0].legend(bbox_to_anchor=(0.5, 0),fontsize = "6",loc='upper center')
    # Remove x-ticks and spines
    splt[0].tick_params(axis='x',          # Changes apply to the x-axis
                        which='both',      # Both major and minor ticks are affected
                        bottom=False,      # Ticks along the bottom edge are off
                        top=False,
                        labeltop=False, # Ticks along the top edge are off
                        labelbottom=False) # Labels along the bottom edge are off

    #splt[0].spines['top'].set_visible(False)    # Top spine is invisible
    #splt[0].spines['bottom'].set_visible(False) # Bottom spine is invisible

    splt[0].title.set_text("Casings")
    
    splt[1].title.set_text("Lithology")
    splt[1].set_ylim([tango,zulu])
    cmap1 = plt.get_cmap('cool')  # Example colormap
    cindex1 = [0,sfs,1,2]
    for item in sorted(cindex1):
        colorvalue1 = (item-0)/4
        splt[1].fill_betweenx(tvd,gr,200,where=lradiff>=item, facecolor=cmap1(colorvalue1))
    if lithos is not None:
        cmap2 = plt.get_cmap('plasma_r')  # Example colormap
        cindex2 = [-0.10,1,2,3]
        for item in sorted(cindex2):
            colorvalue2 = (item-0)/4
            splt[1].fill_betweenx(tvd,-10,gr,where=lithotype>=item, facecolor=cmap2(colorvalue2))
    
    splt[1].plot(gr,tvd,label='GR',color='green',lw=0.25)
    
    
    splt[1].legend(bbox_to_anchor=(0.5, 0),fontsize = "6",loc='upper center')
    splt[1].set_xlim([0,150])
    splt[1].grid(which='both')
    splt[1].grid(which='minor', color = 'teal', alpha = 0.025)
    splt[1].grid(which='major', color = 'teal', alpha = 0.05)
    splt[1].xaxis.set_major_locator(MultipleLocator(50))
    splt[1].xaxis.set_minor_locator(MultipleLocator(10))
    
    #graph.tight_layout()

    # Add your custom plot
    print(mud_weight)
    print(frac_grad_data)
    print(flow_grad_data)
    print(frac_psi_data)
    print(flow_psi_data)
    
    x_values, y_values = zip(*frac_grad_data)
    x_values2, y_values2 = zip(*flow_grad_data)
    x_values3, y_values3 = zip(*frac_psi_data)
    x_values4, y_values4 = zip(*flow_psi_data)
    if UCSs is not None:
        y_values5, x_values5 = zip(*ucss)
        y_values5 = tvdm[find_nearest_depth(md,y_values5)[0]]
    
    #convert y values to tvd
    y_values = tvdm[find_nearest_depth(md,y_values)[0]]
    y_values2 = tvdm[find_nearest_depth(md,y_values2)[0]]
    y_values3 = tvdm[find_nearest_depth(md,y_values3)[0]]
    y_values4 = tvdm[find_nearest_depth(md,y_values4)[0]]
    
    
    #Plot Image
        
    if frac_grad_data != [[0,0]]:
        splt[3].scatter(x_values, y_values, color='dodgerblue', marker='x', s=500)  # Add the custom plot to the second track
    if flow_grad_data != [[0,0]]:
        splt[3].scatter(x_values2, y_values2, color='orange', marker='x', s=500)  # Add the custom plot to the second track
    if frac_psi_data != [[0,0]]:
        splt[4].scatter(x_values3, y_values3, color='dodgerblue', marker='x', s=500)  # Add the custom plot to the second track
    if flow_psi_data != [[0,0]]:
        splt[4].scatter(x_values4, y_values4, color='orange', marker='x', s=500)  # Add the custom plot to the second track
    if UCSs is not None:
        splt[5].scatter(x_values5, y_values5, color='lime', marker='x', s=10)  # Add the custom plot to the second track
    
    #mud_weight_x, mud_weight_y = zip(*mud_weight)
    #splt[1].plot(mud_weight_x, mud_weight_y, color='black', linewidth=2, linestyle='-', drawstyle='steps-post')  # Add the stepped mud_weight line to the second track
    
    #ax2 = splt[2].twinx()
    #ax2.set_ylabel('MD')
    #secax = splt[2].secondary_yaxis('right')
    # Save the modified plot
    plt.gcf().set_size_inches(15, 10)
    plt.savefig(output_file)#,dpi=600)
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
                # You might need to customize this part based on your requirements
                # You can set default values for unit, description, and value or extract from the header if available.
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

