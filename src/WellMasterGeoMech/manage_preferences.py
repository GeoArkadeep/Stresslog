import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import csv

class PreferencesWindow(toga.Window):
    def __init__(self, title, csv1_path, csv2_path, csv3_path, dropdown_options, unitpath):
        super().__init__(title=title)
        self.csv1_path = csv1_path
        self.csv2_path = csv2_path
        self.csv3_path = csv3_path
        self.unitpath = unitpath
        try:
            with open(unitpath, 'r') as f:
                reader = csv.reader(f)
                unitchoice = next(reader)
                unitchoice = [int(x) for x in unitchoice]  # Convert strings to integers
        except:
            unitchoice = [0,0,0,0,0] #Depth, pressure,gradient, strength, temperature
        self.dropdown_options = dropdown_options
        self.unitchoice = unitchoice
        
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
        
        # Create drop-downs
        self.dropdowns = {}
        dropdowns_widgets = []
        for i, (label, options) in enumerate(dropdown_options.items()):
            dropdown_label = toga.Label(label, style=Pack(padding_bottom=5))
            dropdown = toga.Selection(items=options, style=Pack(flex=1, padding_bottom=5))
            
            # Set initial value based on unitchoice
            if i < len(self.unitchoice):
                try:
                    dropdown.value = options[self.unitchoice[i]]
                except IndexError:
                    print(f"Warning: Invalid index {self.unitchoice[i]} for {label}. Using default.")
            
            self.dropdowns[label] = dropdown
            dropdowns_widgets.append(toga.Box(children=[dropdown_label, dropdown], style=Pack(direction=COLUMN, padding_right=5)))
        dropdown_box = toga.Box(children=dropdowns_widgets, style=Pack(direction=ROW, padding_bottom=10))
        
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
        main_box = toga.Box(children=[input_box, dropdown_box, button_box], style=Pack(direction=COLUMN, alignment='center', padding=10, flex=1))
        self.content = main_box
        
    def save_preferences(self, widget):
        # Save contents back to CSV files
        with open(self.csv1_path, 'w') as f:
            f.write(self.csv1_input.value)
        with open(self.csv2_path, 'w') as f:
            f.write(self.csv2_input.value)
        with open(self.csv3_path, 'w') as f:
            f.write(self.csv3_input.value)
        
        # Save unit choices
        unitchoice = []
        for unit_type in ['Depth', 'Pressure', 'Gradient', 'Strength', 'Temperature']:
            value = self.dropdowns[unit_type].value
            options = self.dropdown_options[unit_type]
            try:
                index = options.index(value)
            except ValueError:
                # If the value is not in the list, default to 0
                index = 0
                print(f"Warning: '{value}' not found in options for {unit_type}. Defaulting to first option.")
            unitchoice.append(index)
        
        with open(self.unitpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(unitchoice)
        
        self.close()
    
    def cancel(self, widget):
        self.close()

def show_preferences_window(app, csv1_path, csv2_path, csv3_path, drops, unitpath):
    window = PreferencesWindow("Manage Preferences", csv1_path, csv2_path, csv3_path, drops, unitpath)
    #window.app = app
    app.windows.add(window)
    window.show()