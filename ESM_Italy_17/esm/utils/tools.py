# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:02:21 2021

@author: Amin
"""
import pandas as pd
import random
from typing import List
import xlsxwriter
from collections import namedtuple
import os
import shutil

from esm.utils.constants import _MI, _FORMAT

from esm.log_exc.logging import log_time

import logging

logger = logging.getLogger(__name__)


def dataframe_to_xlsx(path, nans="pass", **kwargs):

    file = xlsxwriter.Workbook(path)
    header_format = file.add_format(_FORMAT)

    for sheet, data in kwargs.items():
        index_levels = data.index.nlevels
        columns_levels = data.columns.nlevels

        rows_start = columns_levels
        cols_start = index_levels

        sheet = file.add_worksheet(sheet)
        for level in range(index_levels):
            rows = data.index.get_level_values(level).to_list()
            counter = 0
            for row in rows:

                sheet.write(rows_start + counter, level, row, header_format)
                counter += 1

        for level in range(columns_levels):
            cols = data.columns.get_level_values(level).to_list()
            counter = 0
            for col in cols:
                sheet.write(level, cols_start + counter, col, header_format)
                counter += 1

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                try:
                    sheet.write(rows_start + row, cols_start + col, data.iloc[row, col])
                except TypeError:
                    if nans == "pass":
                        pass

    file.close()


def dict_to_file(Dict, path, _format="csv",stack=False):
    """Writes nested dicts  to csv"""

    if _format not in ["csv", "xlsx", "txt"]:
        raise ValueError("Acceptable formats are csv ,txt and xlsx.")

    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for key, value in Dict.items():
        if key == "dp":
            continue
        if isinstance(value, pd.DataFrame):
            if stack:
                value = value.unstack(
                    level=[i for i in range(value.index.nlevels)]
                    ).to_frame()

                value.columns = ['value']

            if _format == "xlsx":
                dataframe_to_xlsx(path=f"{path}//{key}.{_format}",Sheet1=value)
                #value.to_excel(f"{path}//{key}.{_format}",merge_cells=merge_cells)
            else:
                value.to_csv(f"{path}//{key}.{_format}")
        else:
            new_path = f"{path}//{key}"
            os.mkdir(new_path)
            dict_to_file(value, new_path, _format,stack=stack)


def delete_duplicates(given, warning=False, comment=None, level="info"):

    """
    This function will delete the duplicated values keeping the order of the values (without sorting them)
    """
    given = list(given)
    unique = list(dict.fromkeys(given))

    if len(given) != len(unique) and warning:
        log_time(logger, comment, level)

    return unique


def remove_items(
    given: [list, pd.Series], item: str = "None", nans: bool = False
) -> list:

    """
    DESCRIPTION
    =============
    This function removes a given item from a list or a pandas.Series

    PARAMETERS
    =============
    given:  what we passed to the function to remove the items
    item : the item to be deleted
    nans : if there is the need to remove the nans from a pd.Series
    """
    if isinstance(given, pd.Series):

        if nans:
            given = list(given.fillna("None"))

        else:
            given = list(given)

    while item in given:
        given.remove(item)

    return given


def generate_random_colors(
    length: int, format: str = "list", based_on_names: [list, None,] = None,
) -> [list, str]:
    """
    DESCRIPTION
    ============
    The function returns random colors

    PARAMETERS
    ============
    length: how many colors to return
    format: how is the format of the outputs -> options: 'list','str'
    based_on_names: if there is a set of names that you want to allocate color
    based on them. In this case, if thare are duplicate names, the colors will be
    duplicate too.

    RETURNS
    ===========
    colors: a list of colors or a single color
    """

    if based_on_names is not None:
        colors = {
            name: "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
            for name in based_on_names
        }

        colors = [colors[name] for name in based_on_names]
    else:
        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
            for i in range(length)
        ]

    if format == "str":
        return colors[0]

    return colors


def cluster_frames(
    instance: object,
    index: pd.MultiIndex,
    columns: pd.MultiIndex,
    default: [int, float],
    parameter: str,
) -> dict:

    """
    The function is in charge of preparing the dataframes of every single
    parameter for the predefined clusters.
    """

    return {
        cluster: pd.DataFrame(default, index=index, columns=columns)
        for cluster in instance.time_cluster.parameter_clusters(parameter)
    }


def excel_read(
    io: str, sheet_name: str, header: list, index_col: list, names: list = None,
) -> pd.DataFrame:

    """
    Description
    ============
    Not different from pd.read_excel. Just specifying the name of the file
    in case of worksheet does not exist.

    """

    try:
        data = pd.read_excel(
            io=io,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            names=names,
        )
    except ValueError as e:
        raise ValueError(f"{io} -> {e}")

    except Exception as e:
        raise Exception(f"{io} -> {e}")

    return data
