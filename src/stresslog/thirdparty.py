import lasio as laua
import io
def datasets_to_las(path, datasets, custom_units={}, **kwargs):
    """
    copyright 2021 Agile Scientific

    license
    Apache 2.0
    """
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
    
    buffer = io.StringIO()
    las.write(buffer, **kwargs)
    buffer.seek(0)  # Move to the beginning of the buffer
    return buffer.getvalue()
    
    # write file to disk
    #with open(path, mode='w') as f:
    #    las.write(f, **kwargs)