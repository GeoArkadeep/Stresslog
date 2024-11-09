import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import csv
import json

class PreferencesWindow(toga.Window):
    def __init__(self, title, csv1_path, csv2_path, csv3_path, dropdown_options, unitpath, algpath):
        super().__init__(title=title)
        self.csv1_path = csv1_path
        self.csv2_path = csv2_path
        self.csv3_path = csv3_path
        self.unitpath = unitpath
        self.algpath = algpath
        
        # Load unit choices
        try:
            with open(unitpath, 'r') as f:
                reader = csv.reader(f)
                unitchoice = next(reader)
                unitchoice = [int(x) for x in unitchoice]
        except:
            unitchoice = [0,0,0,0,0]
            
        # Load algorithm choices
        try:
            with open(algpath, 'r') as f:
                reader = csv.reader(f)
                algchoice = next(reader)
                algchoice = [int(x) for x in algchoice]
        except:
            algchoice = [300,0,0,0,0]  # Default: 300 DPI and first option for each algorithm
            
        self.dropdown_options = dropdown_options
        self.unitchoice = unitchoice
        self.algchoice = algchoice
        
        # Algorithm options
        self.alg_options = {
            'Pore Pressure Algorithm': [
                "Zhang", "Eaton", "Dexp", "Average of All",
                "Zhang > Eaton > Dexp", "Zhang > Dexp > Eaton",
                "Eaton > Zhang > Dexp", "Eaton > Dexp > Zhang",
                "Dexp > Zhang > Eaton", "Dexp > Eaton > Zhang"
            ],
            'Shmin Algorithm': [
                "Daines", "Zoback", "Zoback with dynamic mu"
            ],
            'Shear Failure Algorithm': [
                "Modified Lade", "Mogi-Coulomb", "Mohr-Coulomb"
            ],
            'Downsample Algorithm': [
                "Average", "Median", "Weighted Average"
            ]
        }
        
        # Load CSV contents
        with open(csv1_path, 'r') as f:
            csv1_content = f.read()
        with open(csv2_path, 'r') as f:
            csv2_content = f.read()
        with open(csv3_path, 'r') as f:
            csv3_content = f.read()
        
        # Create multiline inputs
        self.csv1_input = toga.MultilineTextInput(value=csv1_content, style=Pack(flex=1))
        self.csv2_input = toga.MultilineTextInput(value=csv2_content, style=Pack(flex=1))
        self.csv3_input = toga.MultilineTextInput(value=csv3_content, style=Pack(flex=1))
        
        # Create buttons
        save_button = toga.Button('Save', on_press=self.save_preferences, style=Pack(flex=1))
        cancel_button = toga.Button('Cancel', on_press=self.cancel, style=Pack(flex=1))
        
        # Create unit dropdowns
        self.dropdowns = {}
        dropdowns_widgets = []
        for i, (label, options) in enumerate(dropdown_options.items()):
            dropdown_label = toga.Label(label, style=Pack(padding_bottom=5))
            dropdown = toga.Selection(items=options, style=Pack(flex=1, padding_bottom=5))
            
            if i < len(self.unitchoice):
                try:
                    dropdown.value = options[self.unitchoice[i]]
                except IndexError:
                    print(f"Warning: Invalid index {self.unitchoice[i]} for {label}. Using default.")
            
            self.dropdowns[label] = dropdown
            dropdowns_widgets.append(toga.Box(children=[dropdown_label, dropdown], style=Pack(direction=COLUMN, padding_right=5)))
        dropdown_box = toga.Box(children=dropdowns_widgets, style=Pack(direction=ROW, padding_bottom=10))
        
        # Create algorithm dropdowns and DPI input
        self.alg_dropdowns = {}
        alg_widgets = []
        
        # Add algorithm dropdowns
        for i, (label, options) in enumerate(self.alg_options.items()):
            dropdown_label = toga.Label(label, style=Pack(padding_bottom=5))
            dropdown = toga.Selection(items=options, style=Pack(flex=1, padding_bottom=5))
            
            if i < len(self.algchoice) - 1:  # -1 because DPI is first in algchoice
                try:
                    dropdown.value = options[self.algchoice[i + 1]]
                except IndexError:
                    print(f"Warning: Invalid index {self.algchoice[i + 1]} for {label}. Using default.")
            
            self.alg_dropdowns[label] = dropdown
            alg_widgets.append(toga.Box(children=[dropdown_label, dropdown], style=Pack(direction=COLUMN, padding_right=5)))
        
        alg_box = toga.Box(children=alg_widgets, style=Pack(direction=ROW, padding_bottom=10))
        
        # Create DPI input in its own row
        dpi_label = toga.Label('DPI', style=Pack(padding_bottom=5))
        self.dpi_input = toga.NumberInput(style=Pack(flex=1, padding_bottom=5))
        self.dpi_input.value = self.algchoice[0]  # First value is DPI
        dpi_box = toga.Box(
            children=[toga.Box(children=[dpi_label, self.dpi_input], style=Pack(direction=COLUMN))],
            style=Pack(direction=ROW, padding_bottom=10)
        )
        
        # Create layout
        input_box = toga.Box(
            children=[
                toga.Box(children=[toga.Label('Aliases'), self.csv1_input], style=Pack(direction=COLUMN, flex=1)),
                toga.Box(children=[toga.Label('CurveStyles'), self.csv2_input], style=Pack(direction=COLUMN, flex=1)),
                toga.Box(children=[toga.Label('PointStyles'), self.csv3_input], style=Pack(direction=COLUMN, flex=1))
            ],
            style=Pack(direction=ROW, padding=10, flex=1)
        )
        
        button_box = toga.Box(children=[save_button, cancel_button], style=Pack(direction=ROW, padding_top=10, flex=0))
        main_box = toga.Box(
            children=[input_box, dropdown_box, alg_box, dpi_box, button_box],
            style=Pack(direction=COLUMN, alignment='center', padding=10, flex=1)
        )
        self.content = main_box
        
    def save_preferences(self, widget):
        # Save contents back to CSV files
        with open(self.csv1_path, 'w') as f:
            f.write(self.csv1_input.value.replace('\u201c', '"').replace('\u201d', '"'))
        with open(self.csv2_path, 'w') as file:
            json.dump(json.loads(self.csv2_input.value.replace('\u201c', '"').replace('\u201d', '"')), file, indent=4)
        with open(self.csv3_path, 'w') as file:
            json.dump(json.loads(self.csv3_input.value.replace('\u201c', '"').replace('\u201d', '"')), file, indent=4)
        
        # Save unit choices
        unitchoice = []
        for unit_type in ['Depth', 'Pressure', 'Gradient', 'Strength', 'Temperature']:
            value = self.dropdowns[unit_type].value
            options = self.dropdown_options[unit_type]
            try:
                index = options.index(value)
            except ValueError:
                index = 0
                print(f"Warning: '{value}' not found in options for {unit_type}. Defaulting to first option.")
            unitchoice.append(index)
        
        # Save algorithm choices
        algchoice = [int(self.dpi_input.value)]  # Start with DPI value
        for alg_type in ['Pore Pressure Algorithm', 'Shmin Algorithm', 'Shear Failure Algorithm', 'Downsample Algorithm']:
            value = self.alg_dropdowns[alg_type].value
            options = self.alg_options[alg_type]
            try:
                index = options.index(value)
            except ValueError:
                index = 0
                print(f"Warning: '{value}' not found in options for {alg_type}. Defaulting to first option.")
            algchoice.append(index)
        
        # Write choices to files
        with open(self.unitpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(unitchoice)
            
        with open(self.algpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(algchoice)
        
        self.close()
    
    def cancel(self, widget):
        self.close()

def show_preferences_window(app, csv1_path, csv2_path, csv3_path, drops, unitpath, algpath):
    window = PreferencesWindow("Manage Preferences", csv1_path, csv2_path, csv3_path, drops, unitpath, algpath)
    app.windows.add(window)
    window.show()