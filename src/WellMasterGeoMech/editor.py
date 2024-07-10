import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import asyncio
import pandas as pd
import pint
import numpy as np

class CustomEditorWindow(toga.Window):
    def __init__(self, id, title, df, headers, unittypes, unitdict, target_units, datatypes, ureg, on_close_future):
        super().__init__(id=id, title=title)
        
        self.headers = headers
        self.unittypes = unittypes
        self.unitdict = unitdict
        self.target_units = target_units
        self.datatypes = datatypes
        self.ureg = ureg
        self.on_close_future = on_close_future
        self.current_selections = {}
        
        self.initialize_dataframe(df)
        self.create_content()
        
    def load_csv_wrapper(self, widget):
        asyncio.create_task(self.load_csv(widget))

    def initialize_dataframe(self, df):
        if df is None:
            self.df = pd.DataFrame(columns=self.headers)
        else:
            self.df = df.copy()
        
        if list(self.df.columns) != self.headers:
            self.df.columns = self.headers
        
        self.current_units = {header: self.unitdict[unittype][0] for header, unittype in zip(self.headers, self.unittypes)}

    def create_content(self):
        main_box = toga.Box(style=Pack(direction=COLUMN, flex=1))
        
        # Create a box for all content (headers, units, and data)
        self.all_content_box = toga.Box(style=Pack(direction=COLUMN, padding=5))
        
        # Fixed column width
        column_width = 120
        
        # Header row
        header_row = toga.Box(style=Pack(direction=ROW))
        for header in self.headers:
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
        for header, unittype in zip(self.headers, self.unittypes):
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
        
        # Load CSV and Clear Data buttons
        data_buttons = toga.Box(style=Pack(direction=ROW, padding=5))
        load_csv_button = toga.Button('Load CSV', on_press=self.load_csv_wrapper, style=Pack(flex=1))
        clear_data_button = toga.Button('Clear Data', on_press=self.clear_data, style=Pack(flex=1))
        data_buttons.add(load_csv_button)
        data_buttons.add(clear_data_button)
        button_area.add(data_buttons)
        
        # Save and Cancel buttons
        action_buttons = toga.Box(style=Pack(direction=ROW, padding=5))
        save_button = toga.Button('Save', on_press=self.save, style=Pack(flex=1))
        cancel_button = toga.Button('Cancel', on_press=self.cancel, style=Pack(flex=1))
        action_buttons.add(save_button)
        action_buttons.add(cancel_button)
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
        
        # Add data rows
        for i, row in self.df.iterrows():
            data_row = toga.Box(style=Pack(direction=ROW))
            for header in self.headers:
                data_box = toga.Box(style=Pack(direction=COLUMN, width=column_width, padding=2))
                input_box = toga.TextInput(
                    value=str(row[header]), 
                    style=Pack(width=column_width-4, padding=5)
                )
                data_box.add(input_box)
                data_row.add(data_box)
            self.data_box.add(data_row)
        
    def add_row(self, widget):
        new_row = {header: '' for header in self.headers}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.update_data_display()

    def remove_row(self, widget):
        if len(self.df) > 0:
            self.df = self.df.iloc[:-1]
            self.update_data_display()

    def on_current_unit_change(self, widget, header):
        self.current_units[header] = widget.value

    async def load_csv(self, widget):
        try:
            file_path = await self.open_file_dialog(
                title="Select CSV file",
                multiselect=False,
                file_types=['csv']
            )
            if file_path:
                new_df = pd.read_csv(file_path)
                
                # Check if the CSV columns match the expected headers
                if list(new_df.columns) != self.headers:
                    print(f"Warning: CSV columns {list(new_df.columns)} don't match expected headers {self.headers}. Adjusting columns.")
                    # If there are more columns in the CSV than expected, discard extra columns
                    if len(new_df.columns) > len(self.headers):
                        n = len(new_df.columns) - len(self.headers)
                        new_df = new_df.iloc[:,:-n]
                        new_df.reset_index(drop=True,inplace=True)
                        print(f"Discarded extra columns. Using only: {self.headers}")
                    
                    # If there are fewer columns in the CSV than expected
                    elif len(new_df.columns) < len(self.headers):
                        # Add missing columns to the dataframe
                        for header in self.headers[len(new_df.columns):]:
                            new_df[header] = ''
                        print(f"Added missing columns: {self.headers[len(new_df.columns):]}")
                    
                    # Rename the columns to match the headers
                    new_df.columns = self.headers
                
                self.df = new_df
                self.update_data_display()
                
                await self.info_dialog(
                    "Success",
                    "CSV file loaded successfully."
                )
            else:
                print("File selection was canceled.")
        except Exception as e:
            await self.info_dialog(
                "Error",
                f"Failed to load CSV: {str(e)}"
            )
            print(f"Error details: {str(e)}")
        
    def clear_data(self, widget):
        self.on_close_future.set_result(None)
        self.close()

    def save(self, widget):
        # Update dataframe with edited values
        for i, row_box in enumerate(self.data_box.children):
            for j, cell_box in enumerate(row_box.children):
                # The TextInput is the first (and only) child of the cell_box
                input_box = cell_box.children[0]
                self.df.iloc[i, j] = input_box.value
        
        self.df.replace('', np.nan, inplace=True)
        # Perform unit conversion
        for header, current_unit, target_unit in zip(self.headers, self.current_units.values(), self.target_units):
            if current_unit != target_unit:
                try:
                    self.df[header] = self.df[header].apply(lambda x, ureg=self.ureg: ureg.Quantity(float(x), current_unit).to(target_unit).magnitude)
                except:
                    self.df[header] = np.nan
        # Cast columns to required datatypes
        for header, dtype in zip(self.headers, self.datatypes):
            try:
                self.df[header] = self.df[header].astype(dtype)
            except:
                self.df[header] = np.nan
        
        # Drop only rows where all values are NaN
        self.df.dropna(how='all', inplace=True)
        
         # Remove duplicates in the first column, keeping the first occurrence
        first_column = self.headers[0]
        self.df.drop_duplicates(subset=[first_column], keep='first', inplace=True)
        
        #check for empty dataframe
        if self.df.empty:
            self.on_close_future.set_result(None)
            self.close()
        else:
            self.on_close_future.set_result(self.df)
            self.close()

    def cancel(self, widget):
        self.df.replace('', np.nan, inplace=True)
        # Drop only rows where all values are NaN
        self.df.dropna(how='all', inplace=True)
        if self.df.empty:
            self.on_close_future.set_result(None)
            self.close()
        else:
            self.on_close_future.set_result(self.df)
            self.close()

async def custom_edit(app, df, headers, unittypes, unitdict, target_units, datatypes, ureg=pint.UnitRegistry()):
    on_close_future = asyncio.Future()

    editor_window = CustomEditorWindow(
        id='data_editor',
        title='Data Editor',
        df=df,
        headers=headers,
        unittypes=unittypes,
        unitdict=unitdict,
        target_units=target_units,
        datatypes=datatypes,
        ureg=ureg,
        on_close_future=on_close_future
    )
    app.windows.add(editor_window)
    editor_window.show()

    result_df = await on_close_future
    return result_df