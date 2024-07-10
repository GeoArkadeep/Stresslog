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
        main_box = toga.Box(style=Pack(direction=COLUMN))

        # Header labels and unit selection dropdowns at the top
        header_unit_box = toga.Box(style=Pack(direction=COLUMN, padding=(5, 5, 5, 5)))
        
        # Header labels
        header_box = toga.Box(style=Pack(direction=ROW))
        for header in self.headers:
            header_box.add(toga.Label(header, style=Pack(flex=1, text_align='center')))
        header_unit_box.add(header_box)
        
        # Unit selection dropdowns
        unit_box = toga.Box(style=Pack(direction=ROW))
        for header, unittype in zip(self.headers, self.unittypes):
            current_unit = toga.Selection(
                items=self.unitdict[unittype],
                on_change=lambda widget, header=header: self.on_current_unit_change(widget, header),
                style=Pack(flex=1, padding=5)
            )
            self.current_selections[header] = current_unit
            unit_box.add(current_unit)
        
        header_unit_box.add(unit_box)
        #main

        # Scrollable data display and edit area
        self.data_box = toga.Box(style=Pack(direction=COLUMN, padding=(5, 0)))
        self.data_box.add(header_unit_box)
        self.scroll_container = toga.ScrollContainer(content=self.data_box, style=Pack(flex=1))
        self.update_data_display()
        main_box.add(self.scroll_container)

        # Add/Remove row buttons
        row_buttons = toga.Box(style=Pack(direction=ROW, padding=5))
        add_row_button = toga.Button('Add Row', on_press=self.add_row, style=Pack(flex=1))
        remove_row_button = toga.Button('Remove Row', on_press=self.remove_row, style=Pack(flex=1))
        row_buttons.add(add_row_button)
        row_buttons.add(remove_row_button)
        main_box.add(row_buttons)

        # Load CSV and Clear Data buttons
        data_buttons = toga.Box(style=Pack(direction=ROW, padding=5))
        load_csv_button = toga.Button('Load CSV', on_press=self.load_csv_wrapper, style=Pack(flex=1))
        
        clear_data_button = toga.Button('Clear Data', on_press=self.clear_data, style=Pack(flex=1))
        data_buttons.add(load_csv_button)
        data_buttons.add(clear_data_button)
        main_box.add(data_buttons)

        # Save and Cancel buttons
        button_box = toga.Box(style=Pack(direction=ROW, padding=5))
        save_button = toga.Button('Save', on_press=self.save, style=Pack(flex=1))
        cancel_button = toga.Button('Cancel', on_press=self.cancel, style=Pack(flex=1))
        button_box.add(save_button)
        button_box.add(cancel_button)
        main_box.add(button_box)

        self.content = main_box

    def update_data_display(self):
        # Preserve header_unit_box, remove only data rows
        for widget in self.data_box.children[1:]:  # Skip the first child (header_unit_box)
            self.data_box.remove(widget)
        for i, row in self.df.iterrows():
            row_box = toga.Box(style=Pack(direction=ROW, padding=(0, 5)))
            for header in self.headers:
                input_box = toga.TextInput(value=str(row[header]), style=Pack(flex=1, padding=(0, 5)))
                row_box.add(input_box)
            self.data_box.add(row_box)

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
                    print(f"Warning: CSV columns {list(new_df.columns)} don't match expected headers {self.headers}. Renaming columns.")
                    
                    # If the number of columns matches, rename them
                    if len(new_df.columns) == len(self.headers):
                        new_df.columns = self.headers
                    else:
                        # If the number of columns doesn't match, we'll need to handle this case
                        # For now, we'll raise an exception, but you might want to implement a more sophisticated solution
                        raise ValueError(f"Number of columns in CSV ({len(new_df.columns)}) doesn't match expected number of columns ({len(self.headers)})")
                
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

    def clear_data(self, widget):
        self.on_close_future.set_result(None)
        self.close()

    def save(self, widget):
        # Update dataframe with edited values
        for i, row_box in enumerate(self.data_box.children[1:]):  # Skip the first child (header_unit_box)
            for j, input_box in enumerate(row_box.children):
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
        self.df.dropna(inplace=True)
        #check for empty dataframe
        if self.df.empty:
            self.on_close_future.set_result(None)
            self.close()
        else:
            self.on_close_future.set_result(self.df)
            self.close()

    def cancel(self, widget):
        self.df.replace('', np.nan, inplace=True)
        self.df.dropna(inplace=True)
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