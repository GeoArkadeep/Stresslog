import lasio as laua
import io
import numpy as np
import pandas as pd

def datasets_to_las(path, datasets, custom_units={}, **kwargs):
    """
    copyright 2021 Agile Scientific
    license
    Apache 2.0
    """
    """
    Write datasets to a LAS file on disk.

    Args:
        path (Str): Path to write LAS file to. If None, returns string buffer.
        datasets (Dict['<name>': pd.DataFrame]): Dictionary maps a
            dataset name (e.g. 'Curves') or 'Header' to a pd.DataFrame.
        custom_units (Dict[str, str], optional): Dictionary mapping curve names to their units.
            If a curve's unit is not specified, it defaults to an empty string.
    Returns:
        str if path is None, otherwise writes to file
    """
    from functools import reduce
    import warnings
    from lasio import HeaderItem, CurveItem, SectionItems
    
    # ensure no NaN values in header
    if 'Header' in datasets:
        datasets['Header'] = datasets['Header'].fillna('')
    
    # instantiate new LASFile to parse data & header to
    las = laua.LASFile()

    # set header df as variable to later retrieve curve meta data from
    header = datasets['Header']
    
    extracted_units = {}
    if not header.empty:
        curve_header = header[header['section'] == 'Curves']
        for _, row in curve_header.iterrows():
            if row['unit'] and not pd.isna(row['unit']):  # Check for NaN
                extracted_units[str(row['original_mnemonic'])] = str(row['unit'])

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
                    pass  # curves header items are handled in curve data loop
                
                elif section_name == 'Version':
                    for mnem in ['VERS', 'WRAP', 'DLM']:
                        section_data = df_section[df_section.original_mnemonic == mnem]
                        if len(section_data) > 0:
                            setattr(las.version, mnem, str(section_data['value'].values[0]))

                elif section_name in ['Well', 'Parameter']:
                    las.sections[section_name] = SectionItems(
                        [HeaderItem(str(r.original_mnemonic),
                                  str(r.unit) if not pd.isna(r.unit) else '',
                                  str(r.value) if not pd.isna(r.value) else '',
                                  str(r.descr) if not pd.isna(r.descr) else '')
                         for _, r in df_section.iterrows()
                         if not pd.isna(r.original_mnemonic)])  # Skip NaN mnemonics

                elif section_name == 'Other':
                    las.sections["Other"] = str(df_section['descr'].iloc[0]) if not pd.isna(df_section['descr'].iloc[0]) else ''

                else:
                    warnings.warn(f"LAS Section was not recognized: '{section_name}'", stacklevel=2)

        # dataset contains curve data
        if dataset_name in ['Curves', 'Formation', 'Core']:  # explicit curve sections
            for column_name in df.columns:
                curve_data = df[column_name]
                curve_unit = all_units.get(str(column_name), '')
                las.append_curve(mnemonic=str(column_name),
                               data=curve_data,
                               unit=str(curve_unit),
                               descr='',
                               value='')

    # Handle NULL value
    try:
        null_value = float(header[header.original_mnemonic == 'NULL'].value.iloc[0])
    except (IndexError, ValueError):
        null_value = -999.25
    
    # Set the NULL value in both the well section and as general null_value
    las.well['NULL'] = HeaderItem('NULL', '', null_value, 'Null value')
    las.null_value = null_value

    # Set column formatter if not provided
    if 'column_fmt' not in kwargs:
        kwargs['column_fmt'] = column_fmt
        
    # write file to disk or return string
    if path is not None:
        with open(path, mode='w') as f:
            las.write(f, **kwargs)

    buffer = io.StringIO()
    las.write(buffer, **kwargs)
    buffer.seek(0)
    return buffer.getvalue()
