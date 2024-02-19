import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga import Window

import lasio as laua
import welly
#from smoothass import readDevFromAsciiHeader, plotPPmiller
import pandas as pd
import numpy as np

import functools
import os

user_home = os.path.expanduser("~/documents")
app_data = os.getenv("APPDATA")
output_dir = os.path.join(user_home, "pp_app_plots")
output_dir1 = os.path.join(user_home, "pp_app_data")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)

output_file = os.path.join(output_dir, "PlotFigure.png")
output_file2 = os.path.join(output_dir1, "output.csv")
class BackgroundImageView(toga.ImageView):
    def __init__(self, image_path, *args, **kwargs):
        super().__init__(image=toga.Image(image_path), *args, **kwargs)
        self.style.update(flex=1)

#Global Variables
laspath = None
devpath = None
wella = None
deva = None
h1 = None
h2 = None
depth_track = None
attrib = [1,0,0,0,0,0,0,0]

modelfile = open("model.csv", "r") 
# reading the file 
data = modelfile.read()   
# replacing end splitting the text  
# when newline ('\n') is seen. 
data_into_list = data.split(",") 
print(data_into_list) 
modelfile.close() 

#model = np.array([16.33,0.63,0.0008,210,60,0.4,0.8,1,0,2000])
#model = ['16.33','0.63','0.0008','210','60','0.4','0.8','1','0','2000']
model = data_into_list

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
                label = toga.Label(entry_info2['label'], style=Pack(padding_right=5))
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
            
            depth_label = toga.Label("Casing Landing Depth (m)", style=Pack(padding_right=2))
            depth_entry = toga.TextInput(style=Pack(padding_left=5, width=100), initial="0")
            row_box.add(depth_label)
            row_box.add(depth_entry)

            mud_weight_label = toga.Label("Max. Mud Weight", style=Pack(padding_right=2))
            mud_weight_entry = toga.TextInput(style=Pack(padding_left=5, width=100), initial="1")
            row_box.add(mud_weight_label)
            row_box.add(mud_weight_entry)
            
            od_label = toga.Label("Casing OD (inches)", style=Pack(padding_right=2))
            od_entry = toga.TextInput(style=Pack(padding_left=5, width=100), initial="0")
            row_box.add(od_label)
            row_box.add(od_entry)
            
            bitdia_label = toga.Label("Bit Dia (inches)", style=Pack(padding_right=2))
            bitdia_entry = toga.TextInput(style=Pack(padding_left=5, width=100), initial="0")
            row_box.add(bitdia_label)
            row_box.add(bitdia_entry)
            
            iv_label = toga.Label("Casing volume (bbl/100ft)", style=Pack(padding_right=2))
            iv_entry = toga.TextInput(style=Pack(padding_left=5, width=100), initial="0")
            row_box.add(iv_label)
            row_box.add(iv_entry)
            
            ppf_label = toga.Label("Casing Weight (ppf)", style=Pack(padding_right=5))
            ppf_entry = toga.TextInput(style=Pack(padding_left=2, width=100), initial="0")
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
        
        self.page2_btn = toga.Button("Load Data and Proceed", on_press=self.show_page3, style=Pack(padding=10))
        self.page2.add(self.page2_btn)
        
        self.page3 = toga.Box(style=Pack(direction=COLUMN, alignment='center'))
        
        # Create a container with ROW direction for plot and frac_grad_data
        plot_and_data_box = toga.Box(style=Pack(direction=ROW, flex=1))
        
        # Move the frac_grad_data related components inside the plot_and_data_box
        self.frac_grad_data_box = toga.Box(style=Pack(direction=COLUMN, padding_right=10))
        plot_and_data_box.add(self.frac_grad_data_box)
        
        
        # Add the buttons for Add row and Remove row
        row_button_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
        self.add_frac_grad_button = toga.Button("Add Frac Grad", on_press=lambda x: self.add_frac_grad_data_row(x, row_type='frac_grad'), style=Pack(padding=10))
        self.add_frac_psi_button = toga.Button("Add Frac PSI", on_press=lambda x: self.add_frac_grad_data_row(x, row_type='frac_psi'), style=Pack(padding=10))
        row_button_box.add(self.add_frac_grad_button)
        row_button_box.add(self.add_frac_psi_button)
        self.frac_grad_data_box.add(row_button_box)

        # Initialize the list of frac_grad_data rows
        self.frac_grad_data_rows = []
        self.add_frac_grad_data_row(None, row_type='frac_grad')
        self.add_frac_grad_data_row(None, row_type='frac_psi')

        self.bg3 = BackgroundImageView("BG2.png", style=Pack(flex = 5))
        plot_and_data_box.add(self.bg3)
        spacer_box = toga.Box(style=Pack(flex=0.01))  # Add this spacer box
        plot_and_data_box.add(spacer_box)

        # Create a container for flow_grad_data rows
        self.flow_grad_data_box = toga.Box(style=Pack(direction=COLUMN, padding_left=10))
        plot_and_data_box.add(self.flow_grad_data_box)

        # Add the buttons for Add row and Remove row
        flow_row_button_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
        self.add_flow_grad_button = toga.Button("Add Flow Grad", on_press=lambda x: self.add_flow_grad_data_row(x, row_type='flow_grad'), style=Pack(padding=10))
        self.add_flow_psi_button = toga.Button("Add Flow PSI", on_press=lambda x: self.add_flow_grad_data_row(x, row_type='flow_psi'), style=Pack(padding=10))
        flow_row_button_box.add(self.add_flow_grad_button)
        flow_row_button_box.add(self.add_flow_psi_button)
        self.flow_grad_data_box.add(flow_row_button_box)

        # Initialize the list of flow_grad_data rows
        self.flow_grad_data_rows = []
        self.add_flow_grad_data_row(None, row_type='flow_grad')
        self.add_flow_grad_data_row(None, row_type='flow_psi')

        self.page3.add(plot_and_data_box)

        # Define the labels and default values
        global model
        entries_info = [
            {'label': 'RHOA (ppg)', 'default_value': str(model[0])},
            {'label': 'OBG Exp', 'default_value': str(model[1])},
            {'label': 'NCT Exp', 'default_value': str(model[2])},
            {'label': 'DTml (us/ft)', 'default_value': str(model[3])},
            {'label': 'DTmat (us/ft)', 'default_value': str(model[4])},
            {'label': 'Nu', 'default_value': str(model[5])},
            {'label': 'ShaleFlag Cutoff', 'default_value': str(model[6])},
            {'label': 'Window', 'default_value': str(model[7])},
            {'label': 'Start', 'default_value': str(model[8])},
            {'label': 'Stop', 'default_value': str(model[9])},
            {'label': 'WaterDensity', 'default_value': str(model[10])},
            {'label': 'Subhydrostatic', 'default_value': str(model[11])}
            
        ]

        # Create a list to store the textboxes
        self.textboxes = []
        # Add 6 numeric entry boxes with their respective labels
        for i in range(2):
            entry_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
            for j in range(5):
                entry_info = entries_info[5*i+j]
                label = toga.Label(entry_info['label'], style=Pack(padding_right=5))
                entry = toga.TextInput(style=Pack(padding_left=2, width=100))
                entry.value = entry_info['default_value']
                entry_box.add(label)
                entry_box.add(entry)
                self.textboxes.append(entry)
            self.page3.add(entry_box)

        
        button_box3 = toga.Box(style=Pack(direction=ROW, alignment='center', flex=0))
        self.page3_btn1 = toga.Button("Recalculate", on_press=self.get_textbox_values, style=Pack(padding=1))
        button_box3.add(self.page3_btn1)
        
        self.page3_btn2 = toga.Button("Export Plot", on_press=self.show_page1, style=Pack(padding=1))
        button_box3.add(self.page3_btn2)
        
        self.page3_btn3 = toga.Button("Export Las", on_press=self.show_page1, style=Pack(padding=1))
        button_box3.add(self.page3_btn3)
        
        self.page3_btn4 = toga.Button("Back", on_press=self.show_page2, style=Pack(padding=1))
        button_box3.add(self.page3_btn4)
        
        self.page3.add(button_box3)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = self.page1
        self.main_window.show()
        

    def add_frac_grad_data_row(self, widget, row_type='frac_grad'):
        depth_label = toga.Label("Depth", style=Pack(text_align="center", width=100, padding_bottom=5))
        
        if row_type == 'frac_grad':
            second_label = toga.Label("Fracture Gradient", style=Pack(text_align="center", width=100, padding_bottom=5))
        elif row_type == 'frac_psi':
            second_label = toga.Label("Frac PSI", style=Pack(text_align="center", width=100, padding_bottom=5))
        else:
            raise ValueError("Invalid row type")

        depth_input = toga.TextInput(style=Pack(width=100), initial="0")
        second_input = toga.TextInput(style=Pack(width=100), initial="0")

        row_labels = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        row_labels.add(depth_label)
        row_labels.add(second_label)

        row_inputs = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
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
        depth_label = toga.Label("Depth", style=Pack(text_align="center", width=100, padding_bottom=5))
        
        if row_type == 'flow_grad':
            second_label = toga.Label("Flow Gradient", style=Pack(text_align="center", width=100, padding_bottom=5))
        elif row_type == 'flow_psi':
            second_label = toga.Label("Flow PSI", style=Pack(text_align="center", width=100, padding_bottom=5))
        else:
            raise ValueError("Invalid row type")

        depth_input = toga.TextInput(style=Pack(width=100), initial="0")
        second_input = toga.TextInput(style=Pack(width=100), initial="0")

        row_labels = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        row_labels.add(depth_label)
        row_labels.add(second_label)

        row_inputs = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
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
        self.main_window.content = self.page1

    def show_page2(self, widget):
        self.main_window.content = self.page2
    
    def show_page3(self, widget):
        self.set_textbox2_values(widget)
        self.main_window.content = self.page3

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
        depth_mw_values.sort(key=lambda x: x[0])

        return depth_mw_values
    
    def getwelldev(self):
        global laspath, devpath, wella, deva, depth_track
        print(self.dropdown1.value)
        print("Recalculating....   "+str(laspath))
        wella = welly.Well.from_las(laspath, index = "m")
        depth_track = wella.df().index
        if devpath is not None:
            deva=pd.read_csv(devpath, sep=' ',skipinitialspace=True)
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
        #smoothass.plotPPmiller(wella)

        print("Great Success!! :D")
        #image_path = 'PlotFigure.png'
        #self.bg3.image = toga.Image(image_path)


    def on_result0(self, widget, dialog_result):
        global wella, laspath
        if dialog_result:
            laspath = dialog_result
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


    def open_las0(self, widget):
        try:
            self.main_window.open_file_dialog(title="Select a file", multiselect=False, on_result=functools.partial(MyApp.on_result0,self))
        except Exception as e:
            print("Error:", e)

    def on_result1(self, widget, dialog_result):
        global h1, devpath
        if dialog_result:               
            devpath  = dialog_result
            h1 = readDevFromAsciiHeader(devpath)
            print("Loaded dev file:", dialog_result)
            print(h1)
            self.populate_dropdowns()
            self.dropdown1.enabled = True
            self.dropdown2.enabled = True
            self.dropdown3.enabled = True
            self.page1_btn4.enabled = True
            
            
        else:
            print(wella)



    def open_dev0(self, widget):
        try:
            self.main_window.open_file_dialog(title="Select a file", multiselect=False, on_result=functools.partial(MyApp.on_result1,self))
        except Exception as e:
            print("Error:", e)    

    def populate_dropdowns(self):
        global h1
        self.dropdown1.items = h1
        self.dropdown2.items = h1
        self.dropdown3.items = h1
    
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


    
    def get_textbox_values(self,widget):
        global wella
        global attrib
        global model
        self.getwelldev()
        modelfile = open("model.csv", "r") 
        data = modelfile.read()   
        # replacing end splitting the text  
        # when newline (',') is seen. 
        data_into_list = data.split(",") 
        print(data_into_list) 
        modelfile.close() 
        model = data_into_list
        tail1 = str(model[10])
        tail2 = str(model[11])
        tv = [textbox.value for textbox in self.textboxes]
        self.bg3.image = toga.Image('BG1.png')
        model = tv
        model.append(tail1)
        model.append(tail2)
        ih = plotPPmiller(wella,self, float(model[0]), float(model[2]), float(model[1]), float(model[5]), float(model[6]), int(float(model[7])), float(model[8]), float(model[9]), float(model[3]), float(model[4]))
        file = open('model.csv','w')
        for item in model:
            file.write(str(item)+",")
        file.close()
        print("Great Success!! :D")
        image_path = 'PlotFigure.png.png'
        self.bg3.image = toga.Image(output_file)
        self.bg3.refresh()
        
    
    def wellisvertical(self,widget):
        global depth_track
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
        plotPPmiller(wella,self)

        print("Great Success!! :D")
        image_path = 'PlotFigure.png'
        self.bg3.image = toga.Image(output_file)
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

def read_aliases_from_file(file_path='alias.txt'):
    with open(file_path, 'r') as file:
        aliases = eval(file.read())  # Note: Using eval to parse the dictionary from the file
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


def plotPPmiller(well,app_instance, rhoappg = 16.33, lamb=0.0008, a = 0.630, nu = 0.4, sfs = 1.0, window = 1, zulu=0, tango=2000, dtml = 210, dtmt = 60, lala = -1.0, lalb = 1.0, lalm = 5, lale = 0.5, lall = 5, horsuda = 0.77, horsude = 2.93, water = float(model[10]), underbalancereject = bool(model[11]=='True' or model[11] =='true' or model[11]=='TRUE')):
    alias = read_aliases_from_file()
    from welly import Curve
    #print(alias)
    print(well.uwi,well.name)
    #print(well.location.location)
    start_depth = wella.df().index[0]
    final_depth = wella.df().index[-1]
    
    header = well._get_curve_mnemonics()
    print(header)
    alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
    alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
    alias['ssonic'] = [elem for elem in header if elem in set(alias['ssonic'])]
    alias['resdeep'] = [elem for elem in header if elem in set(alias['resdeep'])]
    alias['resshal'] = [elem for elem in header if elem in set(alias['resshal'])]
    alias['density'] = [elem for elem in header if elem in set(alias['density'])]
    alias['neutron'] = [elem for elem in header if elem in set(alias['neutron'])]
    
    detail = app_instance.get_depth_mw_data_values()
    print(detail)
    i = 0
    mud_weight = []
    while i<len(detail):
        mud_weight.append([detail[i][0],detail[i][1]])
        i+=1    
    print(mud_weight)
    first = [mud_weight[0][0],0]
    last = [mud_weight[-1][0],final_depth]
    lastmw = last[0]
    frac_grad_data = app_instance.get_frac_grad_data_values()[0]
    flow_grad_data = app_instance.get_flow_grad_data_values()[0]
    frac_psi_data = app_instance.get_frac_grad_data_values()[1]
    flow_psi_data = app_instance.get_flow_grad_data_values()[1]
    
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
    if alias['ssonic'] != []:
        nu2 = getNu(well, nu)
    else:
        nu2 = [nu] * (len(well.data[alias['sonic'][0]]))
    
    import matplotlib.pyplot as plt
    import math
    gr = well.data[alias['gr'][0]]
    #kth = well.data['KTH']
    dt = well.data[alias['sonic'][0]]
    zden2 = well.data[alias['density'][0]].values
    md = well.data['MD'].values
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
        plt.plot(radiff)
        plt.yscale('log')
        i = 0
        lradiff = np.zeros(len(radiff))
        while i<len(radiff):
            lradiff[i] = radiff[i]
            i+=1
        print("Rdiff :",lradiff)
    
        rediff = Curve(lradiff, mnemonic='ResD',units='ohm/m', index=md, null=0)
        well.data['ResDif'] =  rediff
        print("sfs = :",sfs)
        shaleflag = rediff.block(cutoffs=sfs,values=(0,1)).values
        zoneflag = rediff.block(cutoffs=sfs,values=(0,1)).values
        print(shaleflag)
        plt.plot(shaleflag)        
        plt.show()
    else:
        shaleflag = np.zeros(len(md))
        zoneflag = np.zeros(len(md))
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
    
    glwd = float(attrib[1])
    glf = glwd*(3.28084)
    wdf = glwd*(-3.28084)
    if wdf<0:
        wdf=0
    print(attrib[1])
    well.location.gl = float(attrib[1])
    well.location.kb = float(attrib[0])
    try:
        agf = (well.location.ekb-well.location.egl)*3.28084
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
    i=0
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
        i+=1
    
    print("air gap is ",agf,"feet")
    if glwd>=0:
        print("Ground Level is ",glf,"feet above MSL")
    if glwd<0:
        print("Seafloor is ",wdf,"feet below MSL")
        print(wdfi)
        
    ##print(attrib[1])
    
    rhoppg = np.zeros(len(tvdf))
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
                hydrostatic[i] = water
                mudhydrostatic[i] = 1.0*lastmw
            else:
                if(tvdmsl[i]<0):
                    rhoppg[i] = 8.34540426515252*water
                    hydrostatic[i] = 0
                    mudhydrostatic[i] = 0
                else:
                    rhoppg[i] = 0
                    hydrostatic[i] = water
                    mudhydrostatic[i] = 1.0*lastmw
        else:
            if(tvdbgl[i]>=0):
                rhoppg[i] = rhoappg +(((tvdbglf[i])/3125)**a)
                hydrostatic[i]= water
                mudhydrostatic[i] = 1.0*lastmw
            else:
                rhoppg[i] = 0
                hydrostatic[i] = 0
                mudhydrostatic[i] = 0
        i+=1
    #hydrostatic =  (water*9.80665/6.89476) * tvdmsl
    hydroppf = 0.4335275040012*hydrostatic
    mudppf = 0.4335275040012*mudhydrostatic
    lithostatic =  (2.6*9.80665/6.89476) * tvd
    gradient = lithostatic/(tvdf)*1.48816
    rhoppg[0] = rhoappg
    #rhoppg = interpolate_nan(rhoppg)
    rhogcc =  0.11982642731*rhoppg
#    rhogcc[0] = 0.01
    
    i=1
    maxwaterppg = wdf*8.34540426515252*water
    while(i<len(tvdf-1)):
        if glwd<0:
            if(tvdbgl[i]>0):
                if(tvdmsl[i]>0):
                    ObgTppg[i] =((maxwaterppg + ((rhoppg[i])*(tvdbglf[i])))/tvdmslf[i])
            else:
                if(tvdmsl[i]>0):
                    ObgTppg[i] =(8.34540426515252*water)
        else:
            if (tvdbgl[i]>0):
                ObgTppg[i] =((rhoppg[i])*(tvdbglf[i]))/tvdf[i] #Curved Top Obg Gradient
                #ObgTppg[i] =rhoppg[i] #Flat Top Obg Gradient
        i+=1
    #ObgTppg = interpolate_nan(ObgTppg)
    
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
    if (len(zden2)>10):
        ObgTgcc = [ObgTgcc[i] if math.isnan(zden2[i]) else zden2[i] for i in range(len(zden2))]
        coalflag = [0 if math.isnan(zden2[i]) else 1 if zden2[i]<1.6 else 0 for i in range(len(zden2))]
        lithoflag = [0 if shaleflag[i]<1 else 1 if zden2[i]<1.6 else 2 for i in range(len(zden2))]
    
    
    coal = Curve(coalflag, mnemonic='CoalFlag',units='coal', index=md, null=0)
    litho = Curve(lithoflag, mnemonic='LithoFlag',units='lith', index=md, null=0)
    
    #Millers

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
        obgpsi= ObgTppf[:]*tvdmslf[:]
    else:
        hydropsi = hydroppf[:]*tvdbglf[:]
        obgpsi= ObgTppf[:]*tvdbglf[:]
    mudpsi = mudppf[:]*tvdf[:]
    
    i = 0
    ppgmiller = np.zeros(len(tvdf))
    gccmiller = np.zeros(len(tvdf))
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
    print("Reject Subhydrostatic = ",underbalancereject)
    while i<(len(ObgTppg)-1):
        if glwd>=0:
            if tvdbgl[i]>0:
                if shaleflag[i]<0.5:
                    gccmiller[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    if underbalancereject and gccmiller[i]<1:
                        gccmiller[i]=np.nan
                else:
                    gccmiller[i] = np.nan
                ppgmiller[i] = gccmiller[i]*8.33
                dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                lal3[i] = lall*(304.8/(dalm[i]-1))
                lal[i] = lalm*(vp[i]+lala)/(vp[i]**lale)
                horsud[i] = horsuda*(vp[i]**horsude)
                phi[i] = (vp[i]+lala)/(vp[i]+lalb)
                H[i] = (4*(np.tan(phi[i])**2))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i]))))
                K[i] = (4*lal[i]*(np.tan(phi[i])))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i])))) 
                ym[i] = 0.076*(vp[i]**3.73)
                sm[i] = 0.03*(vp[i]**3.30)
                psiftpp[i] = 0.4335275040012*gccmiller[i]
                psipp[i] = psiftpp[i]*tvdf[i]
                #if psipp[i]<hydropsi[i]:
                #   psipp[i] = hydropsi[i]
        else:
            if tvdbgl[i]>0:
                if shaleflag[i]<0.5:
                    gccmiller[i] = ObgTgcc[i] - ((ObgTgcc[i]-pn)*((math.log((mudline-matrick))-(math.log(dalm[i]-matrick)))/(ct*tvdbgl[i])))
                    if underbalancereject and gccmiller[i]<1:
                        gccmiller[i]=np.nan
                else:
                    gccmiller[i] = np.nan
                ppgmiller[i] = gccmiller[i]*8.33
                dtNormal[i] = matrick + (mudline-matrick)*(math.exp(-ct*tvdbgl[i]))
                lal3[i] = lall*(304.8/(dalm[i]-1))
                lal[i] = lalm*(vp[i]+lala)/(vp[i]**lale)
                horsud[i] = horsuda*(vp[i]**horsude)
                phi[i] = (vp[i]+lala)/(vp[i]+lalb)
                H[i] = (4*(np.tan(phi[i])**2))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i]))))
                K[i] = (4*lal[i]*(np.tan(phi[i])))*(9-(7*np.sin(phi[i])))/(27*(1-(np.sin(phi[i])))) 
                ym[i] = 0.076*(vp[i]**3.73)
                sm[i] = 0.03*(vp[i]**3.30)
                psiftpp[i] = 0.4335275040012*gccmiller[i]
                psipp[i] = psiftpp[i]*tvdf[i]
        i+=1
    #gccmiller[0] = np.nan
    gccmiller[-1] = hydrostatic[-1]
    gccmiller[np.isnan(gccmiller)] = water
    ppgmiller[np.isnan(ppgmiller)] = water*8.33
    psiftpp = interpolate_nan(psiftpp)
    psipp = interpolate_nan(psipp)
    print("GCCmiller: ",gccmiller)
    psiftpp = 0.4335275040012*gccmiller
    #psipp = psiftpp[:]*tvdf[:]
    #ObgTgcc = np.array(ObgTgcc)
    #obgpsift = 0.4335275040012*ObgTgcc
    

    #plt.plot(dtNormal)
    #plt.plot(dalm)
    #plt.show()
    #plt.clf()
    
    #Eatons/Daines
    
    i = 0
    mu = 0.65
    b = 0.0
  
    fgppg = np.zeros(len(ppgmiller))
    fgcc = np.zeros(len(ppgmiller))
    mufgppg = np.zeros(len(ppgmiller))
    mufgcc = np.zeros(len(ppgmiller))

    while i<(len(ObgTppg)-1):
        if tvdbgl[i]>0:
            if shaleflag[i]<0.5:
                fgppg[i] = (nu2[i]/(1-nu2[i]))*(ObgTppg[i]-ppgmiller[i])+ppgmiller[i] +(b*ObgTppg[i])
                mufgppg[i] = ((1/((((mu**2)+1)**0.5)+mu)**2)*(ObgTppg[i]-ppgmiller[i])) + ppgmiller[i]
                mufgcc[i] = 0.11982642731*mufgppg[i]
            else:
                fgppg[i] = np.nan
                mufgppg[i] = np.nan
                mufgcc[i] = np.nan
        fgcc[i] = 0.11982642731*fgppg[i]
        i+=1
    fgppg = interpolate_nan(fgppg)
    fgcc = interpolate_nan(fgcc)
    #fgppg = (nu/(1-nu))(ObgTppg-ppgmiller)+ppgmiller
    psiftfg = 0.4335275040012*fgcc
    
    psifg = psiftfg*tvdf
    psimes = ((psifg+obgpsi)/2)+psipp
    psisfl = (psimes[:]*H[:])+K[:]
    

    #params = {'mnemonic': 'AMOCO', 'run':0, }
    #params={'AMOCO': {'units': 'G/C3'}}
    #data = rhogcc
    #ObgTgcc[0] = np.nan
    
    
    
    
    
    
    #FILTERS
    #Smooth curves using moving averages
    
    i = window
    sfg = fgcc
    while i<len(fgcc):
        sumi = np.sum(fgcc[(i-window):i+(window)])
        sfg[i] = sumi/(2*window)
        i+=1

    
    i = window
    spp = gccmiller
    
    while i<len(gccmiller):
        sumi = np.sum(gccmiller[(i-window):i+(window)])
        spp[i] = sumi/(2*window)
        i+=1
    
    i = window
    spsipp = psipp
    while i<len(psipp):
        sumi = np.sum(psipp[(i-window):i+(window)])
        spsipp[i] = sumi/(2*window)
        i+=1
    i = window
    shorsud = horsud    
    while i<len(horsud):
        sumi = np.sum(horsud[(i-window):i+(window)])
        shorsud[i] = sumi/(2*window)
        i+=1
    i = window
    slal = lal
    while i<len(lal):
        sumi = np.sum(lal[(i-window):i+(window)])
        slal[i] = sumi/(2*window)
        i+=1
    i = window
    slal2 = ym
    while i<len(ym):
        sumi = np.sum(ym[(i-window):i+(window)])
        slal2[i] = sumi/(2*window)
        i+=1
    
    i = window
    slal3 = sm
    while i<len(sm):
        sumi = np.sum(sm[(i-window):i+(window)])
        slal3[i] = sumi/(2*window)
        i+=1

    
    i = window
    spsifp = psifg
    
    while i<len(psifg):
        sumi = np.sum(psifg[(i-window):i+(window)])
        spsifp[i] = sumi/(2*window)
        i+=1
        
    from DrawSP import drawSP       
    doi = 4018
    if doi>0:
        doiactual = find_nearest_depth(tvdm,doi)
        print(doiactual)
        doiA = doiactual[1]
        doiX = doiactual[0]
        print("Depth of interest :",doiA," with index of ",doiX)
        sigmaVmpa = obgpsi[doiX]/145.038
        ppmpa = psipp[doiX]/145.038
        bhpmpa = mudpsi[doiX]/145.038
        ucs = horsud[doiX]
        stresspolygon = [sigmaVmpa,ppmpa,bhpmpa,ucs]
        print(stresspolygon)
        drawSP(sigmaVmpa,ppmpa,bhpmpa,ucs)
    
    
    
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

    pp = Curve(spp, mnemonic='PP_DT_MILLER',units='G/C3', index=md, null=0)
    well.data['PP'] =  pp
    fg = Curve(sfg, mnemonic='FG_DAINES',units='G/C3', index=md, null=0)
    well.data['FG'] =  fg
    fg2 = Curve(mufgcc, mnemonic='FG_ZOBACK',units='G/C3', index=md, null=0)
    well.data['FG2'] =  fg2
    
    
    pppsi = Curve(spsipp, mnemonic='GEOPRESSURE',units='psi', index=md, null=0)
    well.data['PPpsi'] =  pppsi
    fgpsi = Curve(spsifp, mnemonic='FRACTURE_PRESSURE',units='psi', index=md, null=0)
    well.data['FGpsi'] =  fgpsi
    
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
    
    
    graph, (plt1, plt2,plt3,plt4,plt5,plt6,plt7) = plt.subplots(1, 7,sharey=True)
    plt5.plot(slal,tvd,label='lal-direct-c0')
    plt5.plot(slal2,tvd,label='horsud E')
    plt5.plot(slal3,tvd,label='horsud G')
    plt5.plot(shorsud,tvd, label='horsud UCS')
    plt5.invert_yaxis()
    plt5.legend()
    plt5.set_xlim([0,20])
    
    plt1.plot(gr,tvd,color='green',label='GR')
    plt1.legend()
    plt1.set_xlim([0,150])
    
    plt2.plot(dalm,tvd,label='DT')
    plt2.plot(dtNormal,tvd,label='Normal DT (Zhang)')
    plt2.legend()
    plt2.set_xlim([300,50])

    plt3.plot(fg,tvd,color='blue',label='Fracture Gradient (Daines)')
    #plt3.plot(fg2,tvd,color='aqua',label='Fracture Gradient (Zoback)')
    plt3.plot(pp,tvd,color='red',label='Pore Pressure Gradient (Zhang)')
    plt3.plot(obgcc,tvd,color='lime',label='Overburden (Amoco)')
    plt3.legend()
    plt3.set_xlim([0,3])

    plt4.plot(fgpsi,tvd,color='blue',label='Sh min')
    plt4.plot(obgpsi,tvd,color='green',label='Sigma V')
    plt4.plot(hydropsi,tvd,color='aqua',label='Hydrostatic')
    plt4.plot(pppsi,tvd,color='red',label='Pore Pressure')
    plt4.plot(psisfl,tvd,color='pink',label='Lade')
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
    #plt.show()
    plt.clf()
    graph, (plt1, plt2,plt3) = plt.subplots(1, 3,sharey=True)
    graph.suptitle(well.name,fontsize=18)
    """plt5.plot(slal,tvd,label='lal-direct-c0')
    plt5.plot(slal2,tvd,label='horsud E')
    plt5.plot(slal3,tvd,label='horsud G')
    plt5.plot(shorsud,tvd, label='horsud UCS')
    
    plt5.legend()
    plt5.set_xlim([0,20])
    
    
    plt1.plot(gr,tvd,color='green',label='GR')
    plt1.legend()
    plt1.set_xlim([0,150])"""
    
    #plt1.invert_yaxis()
    plt1.set_ylim([tango,zulu])
    plt1.plot(dalm,tvd,label='DT')
    plt1.plot(dtNormal,tvd,label='Normal DT (Zhang)')
    plt1.legend(fontsize = "6",loc='lower center')
    plt1.set_xlim([300,50])
    plt1.title.set_text("Sonic")

    plt2.plot(fg,tvd,color='blue',label='Fracture Gradient (Daines)')
    #plt3.plot(fg2,tvd,color='aqua',label='Fracture Gradient (Zoback)')
    plt2.plot(pp,tvd,color='red',label='Pore Pressure Gradient (Zhang)')
    plt2.plot(obgcc,tvd,color='lime',label='Overburden (Amoco)')
    plt2.legend(fontsize = "6",loc='lower center')
    plt2.set_xlim([0,3])
    plt2.title.set_text("Gradients")

    plt3.plot(fgpsi,tvd,color='blue',label='Sh min')
    plt3.plot(obgpsi,tvd,color='green',label='Sigma V')
    plt3.plot(hydropsi,tvd,color='aqua',label='Hydrostatic')
    plt3.plot(pppsi,tvd,color='red',label='Pore Pressure')
    plt3.plot(mudpsi,tvd,color='pink',label='BHP')
    plt3.legend(fontsize = "6",loc='lower center')
    plt3.title.set_text("Pressures")
    #plt4.set_xlim([0,5000])
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
    #Plot Image
        
    if frac_grad_data != [[0,0]]:
        plt2.scatter(x_values, y_values, color='dodgerblue', marker='x', s=500)  # Add the custom plot to the second track
    if flow_grad_data != [[0,0]]:
        plt2.scatter(x_values2, y_values2, color='orange', marker='x', s=500)  # Add the custom plot to the second track
    
    if frac_psi_data != [[0,0]]:
        plt3.scatter(x_values3, y_values3, color='dodgerblue', marker='x', s=500)  # Add the custom plot to the second track
    if flow_psi_data != [[0,0]]:
        plt3.scatter(x_values4, y_values4, color='orange', marker='x', s=500)  # Add the custom plot to the second track
    
    mud_weight_x, mud_weight_y = zip(*mud_weight)
    plt2.plot(mud_weight_x, mud_weight_y, color='black', linewidth=2, linestyle='-', drawstyle='steps-post')  # Add the stepped mud_weight line to the second track
    
    #ax2 = plt3.twinx()
    #ax2.set_ylabel('MD')
    #secax = plt3.secondary_yaxis('right')

    # Save the modified plot
    plt.gcf().set_size_inches(7, 10)
    plt.savefig(output_file,dpi=300)
    plt.clf()
    return

def readDevFromAsciiHeader(devpath, delim = ' '):
    dev=pd.read_csv(devpath, sep=delim)
    dheader = list(dev.columns)
    return dheader

def main():
    app = MyApp('WellMasterGeoMech', 'com.example.porepressurebuddy')
    return app

if __name__ == "__main__":
    app = MyApp("WellMasterGeoMech", "in.rocklab.porepressurebuddy")
    app.main_loop()

