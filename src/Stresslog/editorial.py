import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
#import asyncio
import pandas as pd
import pint
import numpy as np
import os

class CustomEditorWindow(toga.Window):
    def __init__(self, id, title, csv_path, headers, unittypes, unitdict, target_units, datatypes, ureg):
        super().__init__(id=id, title=title)
        
        self.csv_path = csv_path
        self.expected_headers = headers
        self.unittypes = unittypes
        self.unitdict = unitdict
        self.target_units = target_units
        self.datatypes = datatypes
        self.ureg = ureg
        self.current_selections = {}
        self.data_box = None
        
        self.initialize_dataframe()

    def initialize_dataframe(self):
        if os.path.exists(self.csv_path):
            # Read CSV with first row as headers
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=self.expected_headers)

        # Check if headers match expected headers
        if set(self.df.columns) != set(self.expected_headers):
            self.open_column_mapping_window()
        else:
            self.finalize_dataframe()

    def open_column_mapping_window(self):
        mapping_window = ColumnMappingWindow(self.df.columns, self.expected_headers, self.on_mapping_complete)
        mapping_window.show()

    def on_mapping_complete(self, mapping):
        # Create a new dataframe with mapped columns
        new_df = pd.DataFrame()
        for new_col, old_col in mapping.items():
            if old_col != "None":
                new_df[new_col] = self.df[old_col]
            else:
                new_df[new_col] = np.nan
        
        # Update the dataframe
        self.df = new_df
        
        self.finalize_dataframe()

    def finalize_dataframe(self):
        # Set up current units
        self.current_units = {header: self.unitdict[unittype][0] for header, unittype in zip(self.expected_headers, self.unittypes)}
        
        # Reset index after all operations
        self.df = self.df.reset_index(drop=True)
        
        # Create the content after finalizing the dataframe
        self.create_content()

    def create_content(self):
        main_box = toga.Box(style=Pack(direction=COLUMN, flex=1))
        self.data_box = toga.Box(style=Pack(direction=COLUMN))
        # Create a box for all content (headers, units, and data)
        self.all_content_box = toga.Box(style=Pack(direction=COLUMN, padding=5))
        
        # Fixed column width
        column_width = 120
        
        # Header row
        header_row = toga.Box(style=Pack(direction=ROW))
        for header in self.expected_headers:
            header_box = toga.Box(style=Pack(direction=COLUMN, width=column_width, padding=2))
            header_label = toga.Label(
                header, 
                style=Pack(width=column_width-4, text_align='center', padding=5)
            )
            header_box.add(header_label)
            header_row.add(header_box)
        self.all_content_box.add(header_row)
        
        # Unit selection row
        unit_row = toga.Box(style=Pack(direction=ROW))
        for header, unittype in zip(self.expected_headers, self.unittypes):
            unit_box = toga.Box(style=Pack(direction=COLUMN, width=column_width, padding=2))
            current_unit = toga.Selection(
                items=self.unitdict[unittype],
                on_change=lambda widget, header=header: self.on_current_unit_change(widget, header),
                style=Pack(width=column_width-4, padding=5)
            )
            self.current_selections[header] = current_unit
            unit_box.add(current_unit)
            unit_row.add(unit_box)
        self.all_content_box.add(unit_row)
        
        # Data display area (will be populated in update_data_display)
        self.data_box = toga.Box(style=Pack(direction=COLUMN))
        self.all_content_box.add(self.data_box)
        
        # Put all content in a scroll container
        self.scroll_container = toga.ScrollContainer(content=self.all_content_box, style=Pack(flex=1))
        main_box.add(self.scroll_container)
        
        # Button area
        button_area = toga.Box(style=Pack(direction=COLUMN))
        
        # Add/Remove row buttons
        row_buttons = toga.Box(style=Pack(direction=ROW, padding=5))
        add_row_button = toga.Button('Add Row', on_press=self.add_row, style=Pack(flex=1))
        remove_row_button = toga.Button('Remove Row', on_press=self.remove_row, style=Pack(flex=1))
        row_buttons.add(add_row_button)
        row_buttons.add(remove_row_button)
        button_area.add(row_buttons)
        
        # Copy to Clipboard and Paste Data buttons
        data_buttons = toga.Box(style=Pack(direction=ROW, padding=5))
        copy_clipboard_button = toga.Button('Copy to Clipboard', on_press=self.copy_to_clipboard, style=Pack(flex=1))
        paste_data_button = toga.Button('Paste from Clipboard', on_press=self.load_from_clipboard, style=Pack(flex=1))
        data_buttons.add(copy_clipboard_button)
        data_buttons.add(paste_data_button)
        button_area.add(data_buttons)
        
        # Save and Clear buttons
        action_buttons = toga.Box(style=Pack(direction=ROW, padding=5))
        save_button = toga.Button('Save', on_press=self.save, style=Pack(flex=1))
        clear_button = toga.Button('Clear', on_press=self.clear_data, style=Pack(flex=1))
        action_buttons.add(save_button)
        action_buttons.add(clear_button)
        button_area.add(action_buttons)
        
        main_box.add(button_area)
        
        self.content = main_box
        
        # Update data display after setting up the structure
        self.update_data_display()

    def update_data_display(self):
        # Clear existing data rows
        for child in list(self.data_box.children):
            self.data_box.remove(child)
        
        # Fixed column width
        column_width = 120
        
        # Create all rows at once, with a maximum of 1000 rows to prevent memory issues
        all_rows = []
        for i, row in self.df.head(100).iterrows():
            data_row = self.create_data_row(row, column_width)
            all_rows.append(data_row)
        
        # Add all rows to the data_box at once
        self.data_box.add(*all_rows)

        # If there are more than 1000 rows, inform the user
        if len(self.df) > 100:
            print(f"Warning: Only displaying first 1000 rows out of {len(self.df)} total rows.")

    def create_data_row(self, row, column_width):
        data_row = toga.Box(style=Pack(direction=ROW))
        for header in self.expected_headers:
            data_box = toga.Box(style=Pack(direction=COLUMN, width=column_width, padding=2))
            input_box = toga.TextInput(
                value=str(row[header]), 
                style=Pack(width=column_width-4, padding=5)
            )
            data_box.add(input_box)
            data_row.add(data_box)
        return data_row
        
    def add_row(self, widget):
        new_row = {header: '' for header in self.expected_headers}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Create and add only the new row
        new_data_row = self.create_data_row(new_row, 120)
        self.data_box.add(new_data_row)

    def remove_row(self, widget):
        if len(self.df) > 0:
            self.df = self.df.iloc[:-1]
            
            # Remove only the last row from the UI
            if self.data_box.children:
                self.data_box.remove(self.data_box.children[-1])

    def on_current_unit_change(self, widget, header):
        self.current_units[header] = widget.value

    def copy_to_clipboard(self, widget):
        try:
            self.df.to_clipboard(index=False)
            self.main_window.info_dialog("Success", "Data copied to clipboard successfully.")
        except Exception as e:
            self.main_window.error_dialog("Error", f"Failed to copy data to clipboard: {str(e)}")
            print(f"Error details: {str(e)}")

            
        
    def clear_data(self, widget):
        os.remove(self.csv_path) if os.path.exists(self.csv_path) else None
        self.close()

    def load_from_clipboard(self, widget):
        try:
            new_df = pd.read_clipboard(sep='\s+')
            self.process_loaded_data(new_df, "Clipboard")
            self.info_dialog("Success", "Data loaded from clipboard successfully.")
        except Exception as e:
            self.error_dialog("Error", f"Failed to load data from clipboard: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def process_loaded_data(self, new_df, source):
        print(f"Processing {source} data...")
        print(f"{source} columns: {list(new_df.columns)}")
        print(f"Expected headers: {self.expected_headers}")

        # Create a new dataframe with the expected columns
        processed_df = pd.DataFrame(columns=self.expected_headers)

        # Copy data from new_df to processed_df column by column
        for i, header in enumerate(self.expected_headers):
            if i < len(new_df.columns):
                processed_df[header] = new_df.iloc[:, i]
            else:
                processed_df[header] = ''  # Add blank column if source has fewer columns
                print(f"Added missing column: {header}")

        # If new_df has more columns than expected, print a warning
        if len(new_df.columns) > len(self.expected_headers):
            print(f"Warning: {source} has {len(new_df.columns) - len(self.expected_headers)} extra columns that were not loaded.")

        # Update the dataframe and display
        self.df = processed_df
        self.update_data_display()
    
    def save(self, widget):
        # Update dataframe with edited values
        for i, row_box in enumerate(self.data_box.children):
            for j, cell_box in enumerate(row_box.children):
                # The TextInput is the first (and only) child of the cell_box
                input_box = cell_box.children[0]
                self.df.iloc[i, j] = input_box.value
        
        self.df.replace('NaN', np.nan, inplace=True)
        self.df.replace("nan", np.nan, inplace=True)
        self.df.replace('', np.nan, inplace=True)
        # Perform unit conversion
        for header, current_unit, target_unit in zip(self.expected_headers, self.current_units.values(), self.target_units):
            if current_unit != target_unit:
                try:
                    self.df[header] = self.df[header].apply(lambda x, ureg=self.ureg: ureg.Quantity(float(x), current_unit).to(target_unit).magnitude)
                except:
                    self.df[header] = np.nan
        # Cast columns to required datatypes
        for header, dtype in zip(self.expected_headers, self.datatypes):
            try:
                self.df[header] = self.df[header].astype(dtype)
            except:
                self.df[header] = np.nan
        
        # Drop only rows where all values are NaN
        self.df.dropna(how='all', inplace=True)
        
         # Remove duplicates in the first column, keeping the first occurrence
        first_column = self.expected_headers[0]
        self.df.drop_duplicates(subset=[first_column], keep='first', inplace=True)
        
        # Drop rows where the first column is NaN
        self.df = self.df[~(self.df[first_column].isna() | 
                    (self.df[first_column].astype(str).str.strip() == '0') | 
                    (self.df[first_column] == 0))]
        
        # Sort the dataframe based on the first column in ascending order
        self.df.sort_values(by=[first_column], inplace=True)
        
        # Save the dataframe back to the CSV file
        if not self.df.empty:
            self.df.to_csv(self.csv_path, index=False)
        
        self.close()

    def cancel(self, widget):
        self.close()

class ColumnMappingWindow(toga.Window):
    def __init__(self, current_headers, expected_headers, on_complete_callback):
        super().__init__(title="Map Columns")
        self.current_headers = ["None"] + list(current_headers)
        self.expected_headers = expected_headers
        self.on_complete_callback = on_complete_callback
        self.mapping = {}
        self.create_content()

    def create_content(self):
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
        
        for expected_header in self.expected_headers:
            row = toga.Box(style=Pack(direction=ROW, padding=5))
            label = toga.Label(expected_header, style=Pack(flex=1))
            dropdown = toga.Selection(items=self.current_headers, style=Pack(flex=1))
            row.add(label)
            row.add(dropdown)
            main_box.add(row)
            self.mapping[expected_header] = dropdown

        button = toga.Button("Confirm Mapping", on_press=self.confirm_mapping)
        main_box.add(button)

        self.content = main_box

    def confirm_mapping(self, widget):
        mapping = {expected: dropdown.value for expected, dropdown in self.mapping.items()}
        self.on_complete_callback(mapping)
        self.close()

def custom_edit(app, csv_path, headers, unittypes, unitdict, target_units, datatypes, ureg=pint.UnitRegistry()):
    editor_window = CustomEditorWindow(
        id='data_editor',
        title='Data Editor',
        csv_path=csv_path,
        headers=headers,
        unittypes=unittypes,
        unitdict=unitdict,
        target_units=target_units,
        datatypes=datatypes,
        ureg=ureg
    )
    app.windows.add(editor_window)
    editor_window.show()