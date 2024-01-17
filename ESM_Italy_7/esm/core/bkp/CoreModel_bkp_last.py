# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:36:56 2021

@author: Amin
"""
from esm.utils.constants import (
    _SETS_READ,
    _MI,
    _CLUSTER_YEARLY_SHEETS,
    _CLUSTER_SLICE_SHEETS,
    _AV_MAP,
    _COST_ITEMS,
)

from esm.utils.errorCheck import (
    check_excel_col_row,
    validate_data,
    check_file_extension,
    nans_exist,
)

from esm.utils.tools import (
    delete_duplicates,
    remove_items,
    generate_random_colors,
    dataframe_to_xlsx,
    excel_read,
    dict_to_file,
)

from esm.log_exc.exceptions import (
    WrongInput,
    WrongExcelFormat,
)

from esm.utils import cvxpyModified as cm
from esm.log_exc.logging import log_time
from esm.core.Properties import Identifiers
from esm.core.Clusters import NonTimeCluster, TimeCluster
from esm.core.FreezeDict import Dict

import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import logging
from copy import deepcopy as dc

logger = logging.getLogger(__name__)


class Base(Identifiers):
    """
    DESCRIPTION
    ============
    Base class, provides the basic methods and properties of the energy model.
    The user will not instanciate this class.
    This class is created as a parent class for the Model class
    """

    def __readsets__(self, path):

        """
        DESCRIPTION
        ============
        This function is in charge of reading all the sets based on the predifend structure
        by _SETS_READ and sets the attrubutres.

        """
        if isinstance(path, str):
            # Read the sets from excel file
            self._read_sets_from_excel(path)

        self.__set_properties__()
        self.__Indexer__()
        self.time_cluster = TimeCluster(self)
        self.__Frames__()
        self.data = dc(dict(**self.__matrices__))

    def _item_indexer(self, item):
        """
        Function returns some infromation regarding the index levels
        when reading excel files.
        """

        if item == "TechnologyData":
            index = self.indeces["u"]["index"]
            columns = self.indeces["u"]["columns"]
            matrices = [
                "u",
                "bp",
                "wu",
                "bu",
                "st",
                "cu",
                "tp",
            ]
            ef = [f"ef_{flow}" for flow in self.ids.emission_by_flows]
            matrices.extend(ef)

        elif item == "DemandProfiles":
            index = self.indeces["dp"]["index"]
            columns = self.indeces["dp"]["columns"]
            matrices = ["dp"]

        elif item == "MoneyRates":
            index = self.indeces["mr"]["index"]
            columns = self.indeces["mr"]["columns"]
            matrices = ["mr"]
        elif item == "FlowData":
            index = self.indeces["v"]["index"]
            columns = self.indeces["v"]["columns"]
            matrices = [
                "v",
                "bv",
            ]

        elif item == "Demand":
            index = self.indeces["E"]["index"]
            columns = self.indeces["E"]["columns"]
            matrices = self.ids.consumption_sectors

        elif item == "DemandCoefficients":
            index = self.indeces["e"]["index"]
            columns = self.indeces["e"]["columns"]
            matrices = [
                "e",
            ]

        elif item == "Availability":
            all_avaliabilities = pd.concat(
                [
                    pd.DataFrame(index=["dummy"], columns=self.indeces[item]["columns"])
                    for item in ["af_eq", "af_min", "af_max"]
                ],
                axis=1,
            )

            # the three levels of eq,min and max have the same index
            index = self.indeces["af_eq"]["index"]
            columns = all_avaliabilities.columns
            matrices = ["af_eq", "af_min", "af_max"]

        elif item == "TechProductionMix":
            index = self.indeces["xy_mix"]["index"]
            columns = self.indeces["xy_mix"]["columns"]
            matrices = ["xy_mix"]

        elif item == "OperativeCapacityMax":
            index = self.indeces["cap_o_max"]["index"]
            columns = self.indeces["cap_o_max"]["columns"]
            matrices = ["cap_o_max"]

        if isinstance(columns, pd.MultiIndex):
            col_range = len(columns[0])
        else:
            col_range = 1

        if isinstance(index, pd.MultiIndex):
            ind_range = len(index[0])
        else:
            ind_range = 1

        return dict(
            index=index,
            columns=columns,
            header=[i for i in range(col_range)],
            index_col=[i for i in range(ind_range)],
            matrices=matrices,
        )

    def _read_paramaters_from_excel(
        self,
        file: str,
        item: str,
    ) -> None:
        """
        Reading the inputs from excel file
        """
        all_clusters = self.time_cluster.parameter_clusters("all")
        __indexer__ = self._item_indexer(item)

        if item in ["Demand", "DemandCoefficients"]:
            to_fill = self.data["E"]
            for sheet in __indexer__["matrices"]:
                data = excel_read(
                    io=file,
                    sheet_name=sheet,
                    header=__indexer__["header"],
                    index_col=__indexer__["index_col"],
                )

                check_excel_col_row(
                    given=data.index,
                    correct=__indexer__["index"],
                    file=file,
                    sheet=sheet,
                    level="index",
                    check="equality",
                )

                check_excel_col_row(
                    given=data.columns,
                    correct=__indexer__["columns"],
                    file=file,
                    sheet=sheet,
                    level="columns",
                    check="equality",
                )

                index = to_fill.index
                columns = self.ids.all_years

                to_fill.loc[index, (sheet, columns)] = data.loc[
                    (index.get_level_values(0), index.get_level_values(1), slice(None)),
                    columns,
                ].values

        elif item == "MoneyRates":
            data = excel_read(
                io=file,
                sheet_name=0,
                header=__indexer__["header"],
                index_col=__indexer__["index_col"],
            )
            # checking if the columsn are correct for every sheet to read
            check_excel_col_row(
                given=data.columns,
                correct=__indexer__["columns"],
                file=file,
                sheet=0,
                level="columns",
                check="equality",
            )
            check_excel_col_row(
                given=data.index,
                correct=__indexer__["index"],
                file=file,
                sheet=0,
                level="index",
                check="equality",
            )

            to_fill = self.data["mr"]
            index = to_fill.index
            columns = to_fill.columns

            self.data["mr"].loc[index, columns] = data.loc[index, columns].values

        elif item == "DemandProfiles":
            for product in self.ids.hourly_products:
                for cluster in self.time_cluster.parameter_clusters("dp"):

                    sheet_name = f"{product}.{cluster}"
                    data = excel_read(
                        io=file,
                        sheet_name=sheet_name,
                        header=__indexer__["header"],
                        index_col=__indexer__["index_col"],
                    )

                    # checking if the columsn are correct for every sheet to read
                    check_excel_col_row(
                        given=data.columns,
                        correct=__indexer__["columns"],
                        file=file,
                        sheet=sheet_name,
                        level="columns",
                        check="equality",
                    )

                    data.columns = [data.columns.get_level_values(i) for i in [0, 1]]

                    to_fill = self.data["dp"][cluster][product]
                    index = to_fill.index
                    columns = to_fill.columns

                    try:
                        self.data["dp"][cluster][product].loc[
                            index, columns
                        ] = data.loc[index, columns].values
                    except KeyError:
                        raise WrongExcelFormat(
                            "Possible issues found in the indexing in {}, sheet {}. "
                            "To avoid problems it is suggested to use the built-in functions "
                            "to print the input files.".format(file, sheet_name)
                        )

        elif item == "TechProductionMix":
            frame = pd.DataFrame()
            to_fill = self.data["xy_mix"]
            for region in self.Regions:
                data = excel_read(
                    io=file,
                    sheet_name=region,
                    header=__indexer__["header"],
                    index_col=__indexer__["index_col"],
                    names=None,
                )
                # avoiding assignement of index if first row is nan
                if data.index.name:
                    row = data.index.name
                    data.index.name = None
                    data.loc[row, :] = np.nan
                    data = data.sort_index()

                check_excel_col_row(
                    given=data.columns,
                    correct=__indexer__["columns"],
                    file=file,
                    sheet=region,
                    level="columns",
                    check="equality",
                )

                index = to_fill.index

                columns_0 = data.columns.get_level_values(0).tolist()
                columns_1 = data.columns.get_level_values(1).tolist()
                try:
                    to_fill.loc[index, (region, columns_0, columns_1)] = data.loc[
                        index, (columns_0, columns_1)
                    ].values

                except KeyError as e:
                    raise WrongExcelFormat(
                        "Model can not find {} in {}, sheet {}"
                        ". This can be due to nan values in the excel file.".format(
                            e.args, file, sheet_name
                        )
                    )

        elif item == "OperativeCapacityMax":
            frame = pd.DataFrame()
            to_fill = self.data["cap_o_max"]
            for region in self.Regions:
                data = excel_read(
                    io=file,
                    sheet_name=region,
                    header=__indexer__["header"],
                    index_col=__indexer__["index_col"],
                    names=None,
                )

                # avoiding assignement of index if first row is nan
                if data.index.name:
                    row = data.index.name
                    data.index.name = None
                    data.loc[row, :] = np.nan
                    data = data.sort_index()

                check_excel_col_row(
                    given=data.columns,
                    correct=__indexer__["columns"],
                    file=file,
                    sheet=region,
                    level="columns",
                    check="equality",
                )

                index = to_fill.index

                columns_0 = data.columns.get_level_values(0).tolist()
                columns_1 = data.columns.get_level_values(1).tolist()
                try:
                    to_fill.loc[index, (region, columns_0, columns_1)] = data.loc[
                        index, (columns_0, columns_1)
                    ].values

                except KeyError as e:
                    raise WrongExcelFormat(
                        "Model can not find {} in {}, sheet {}"
                        ". This can be due to nan values in the excel file.".format(
                            e.args, file, sheet_name
                        )
                    )

        else:
            for region in getattr(self, _MI["r"]):
                for cluster in all_clusters:
                    sheet_name = f"{region}.{cluster}"

                    data = excel_read(
                        io=file,
                        sheet_name=sheet_name,
                        header=__indexer__["header"],
                        index_col=__indexer__["index_col"],
                    )

                    for matrix in __indexer__["matrices"]:
                        # search for the data of the matrix if it exist in the cluster
                        if cluster in self.time_cluster.parameter_clusters(matrix):
                            # checking if the columns are correct for every sheet to read
                            check_excel_col_row(
                                given=data.columns,
                                correct=__indexer__["columns"],
                                file=file,
                                sheet=sheet_name,
                                level="columns",
                                check="equality",
                            )
                            if item == "Availability":
                                try:
                                    take = data[_AV_MAP[matrix]]
                                except KeyError as e:
                                    raise WrongExcelFormat(
                                        "Model can not find {} in {}, sheet {}"
                                        ". This can be due to nan values in the excel file.".format(
                                            e.args, file, sheet_name
                                        )
                                    )
                                take.columns = take.columns.get_level_values(1)

                                to_fill = self.data[matrix][cluster]

                                index = to_fill.index
                                columns = delete_duplicates(
                                    to_fill.columns.get_level_values(-1)
                                )

                                try:
                                    self.data[matrix][cluster].loc[
                                        index, (region, columns)
                                    ] = take.loc[index, columns].values

                                except KeyError:
                                    raise WrongExcelFormat(
                                        "Possible issues found in the indexing in {}, sheet {} for matrix {}. "
                                        "To avoid problems it is suggested to use the built-in functions "
                                        "to print the input files.".format(
                                            file, sheet_name, matrix
                                        )
                                    )

                            elif item in ["TechnologyData", "FlowData"]:

                                try:
                                    take = data.loc[matrix, :]
                                except KeyError as e:
                                    raise WrongExcelFormat(
                                        "Model can not find {} in {}, sheet {}"
                                        ". This can be due to nan values in the excel file.".format(
                                            e.args, file, sheet_name
                                        )
                                    )

                                # we need to change the index format to be inline
                                # with what we have in __matrices__
                                if item == "TechnologyData":
                                    index_level = 0
                                    take.index = take.index.get_level_values(
                                        index_level
                                    )
                                    take.columns = take.columns.get_level_values(1)

                                    if matrix[0:2] == "ef":
                                        to_fill = self.data["ef"][cluster][matrix[3:]]
                                    else:
                                        to_fill = self.data[matrix][cluster]

                                    index = delete_duplicates(
                                        to_fill.index.get_level_values(-1)
                                    )
                                    columns = delete_duplicates(
                                        to_fill.columns.get_level_values(-1)
                                    )

                                    if matrix in ["bp", "wu", "bu", "st", "cu", "tp"]:
                                        indexer = index
                                    else:
                                        indexer = (region, index)

                                else:
                                    if matrix == "v":
                                        take.index = [
                                            take.index.get_level_values(i)
                                            for i in [0, 1]
                                        ]
                                    else:
                                        take.index = take.index.get_level_values(1)

                                    take.columns = take.columns.get_level_values(0)

                                    to_fill = self.data[matrix][cluster]

                                    index = to_fill.index
                                    indexer = index
                                    columns = delete_duplicates(
                                        to_fill.columns.get_level_values(-1)
                                    )

                                try:
                                    if matrix[0:2] == "ef":

                                        self.data["ef"][cluster][matrix[3:]].loc[
                                            indexer, (region, columns)
                                        ] = take.loc[index, columns].values
                                    else:
                                        self.data[matrix][cluster].loc[
                                            indexer, (region, columns)
                                        ] = take.loc[index, columns].values
                                except KeyError:
                                    raise WrongExcelFormat(
                                        "Possible issues found in the indexing in {}, sheet {} for matrix {}. "
                                        "To avoid problems it is suggested to use the built-in functions "
                                        "to print the input files.".format(
                                            file, sheet_name, matrix
                                        )
                                    )

    def _read_sets_from_excel(self, path: str) -> None:

        """
        This function will be used in read __readsets__ function if the given
        path is a str.

        """

        self.warnings = []
        self.non_time_clusters = NonTimeCluster()

        sets_frames = {}
        for set_name, info in _SETS_READ.items():

            data = pd.read_excel(path, **info["read"])

            log_time(logger, f"Sets: {set_name} sheet imported successfully.")

            check_excel_col_row(
                given=list(data.columns),
                correct=info["columns"],
                file=path,
                sheet=info["read"]["sheet_name"],
                level="columns",
                check="contain",
            )

            for validation_item, acceptable_values in info["validation"].items():

                if isinstance(acceptable_values, str):
                    acceptable_values = eval(acceptable_values)

                validate_data(
                    list(data[validation_item]),
                    acceptable_values,
                    f"{set_name}, column:{validation_item}",
                )

            for non_acceptable_nan_columns in info["stop_nans"]:
                nans_exist(
                    data=data[non_acceptable_nan_columns],
                    action="raise error",
                    info=f"{set_name}: {non_acceptable_nan_columns}",
                )

            sets_without_nans = remove_items(data[info["set"]], nans=True)
            sets_unique = delete_duplicates(
                sets_without_nans,
                warning=True,
                comment=f"{set_name} has duplciate values"
                " in the rows. only the first row of"
                " duplicate values will be kept.",
                level="critical",
            )

            sets_sorted = sorted(sets_unique) if info["sort"] else sets_unique

            # Filling the default values
            data = self.__default_values__(
                data=data,
                category="sets",
                info=info,
                name=set_name,
            )

            # Setting the index of dataframe based on unique sets_unique
            data = data.drop_duplicates(subset=info["set"])
            data = data.set_index([info["set"]])

            data = data.loc[sets_sorted, :]

            if set_name == _MI["h"] and data.shape[0] != 1:
                raise WrongInput(
                    "for {}, only one item (row) can be defined.".format(_MI["h"])
                )

            if set_name != _MI["h"]:
                self.non_time_clusters.check_cluster_exists(
                    dataframe=data, set_name=set_name
                )

                data = self.non_time_clusters.re_organize_main_dataframes(set_name)

            sets_frames[set_name] = data
            setattr(self, set_name, sets_sorted)

            log_time(logger, f"Sets: {set_name} creted successfully")

        self.__sets_frames__ = Dict(**sets_frames)

    def __generate_excel__(self, path, what: str):

        """
        This function generates formatted excel input files

        """

        write = True
        sheets = {}
        all_clusters = self.time_cluster.parameter_clusters("all")

        if what == "TechnologyData":

            for region in self.Regions:
                for cluster in all_clusters:

                    frame = pd.DataFrame()
                    sheet_name = f"{region}.{cluster}"

                    for item in _CLUSTER_YEARLY_SHEETS:

                        if item in [
                            "v",
                            "bv",
                            "E",
                            "m",
                            "e",
                        ]:
                            continue
                        if cluster in self.time_cluster.parameter_clusters(item):

                            new_frame = pd.DataFrame(
                                data=0,
                                index=self.indeces[item]["index"],
                                columns=self.indeces[item]["columns"],
                            )

                            frame = pd.concat([frame, new_frame])

                    sheets[sheet_name] = frame

        elif what == "TechProductionMix":

            for region in self.Regions:

                sheets[region] = pd.DataFrame(
                    data=0,
                    index=self.indeces["xy_mix"]["index"],
                    columns=self.indeces["xy_mix"]["columns"],
                )

        elif what == "OperativeCapacityMax":

            for region in self.Regions:

                sheets[region] = pd.DataFrame(
                    data=0,
                    index=self.indeces["cap_o_max"]["index"],
                    columns=self.indeces["cap_o_max"]["columns"],
                )

        elif what == "FlowData":

            for region in self.Regions:
                for cluster in all_clusters:

                    frame = pd.DataFrame()
                    sheet_name = f"{region}.{cluster}"

                    for item in ["v", "bv"]:

                        if cluster in self.time_cluster.parameter_clusters(item):

                            new_frame = pd.DataFrame(
                                data=0,
                                index=self.indeces[item]["index"],
                                columns=self.indeces[item]["columns"],
                            )
                            frame = pd.concat([frame, new_frame])

                    sheets[sheet_name] = frame

        elif what == "DemandProfiles":

            for product in self.ids.hourly_products:
                for cluster in all_clusters:
                    sheet_name = f"{product}.{cluster}"
                    if cluster in self.time_cluster.parameter_clusters("dp"):
                        frame = pd.DataFrame(
                            data=0,
                            index=self.indeces["dp"]["index"],
                            columns=self.indeces["dp"]["columns"],
                        )

                    sheets[sheet_name] = frame

        elif what == "Availability":

            for region in self.Regions:
                for cluster in all_clusters:

                    frame = pd.DataFrame()
                    sheet_name = f"{region}.{cluster}"

                    for item in _CLUSTER_SLICE_SHEETS:
                        if (
                            cluster in self.time_cluster.parameter_clusters(item)
                            and item != "dp"
                        ):
                            new_frame = pd.DataFrame(
                                data=0,
                                index=self.indeces[item]["index"],
                                columns=self.indeces[item]["columns"],
                            )

                            frame = pd.concat([frame, new_frame], axis=1)

                    sheets[sheet_name] = frame

        elif what == "Demand":
            if self.mode == "stand-alone":

                for sector in self.ids.consumption_sectors:
                    sheets[sector] = pd.DataFrame(
                        data=0,
                        index=self.indeces["E"]["index"],
                        columns=self.indeces["E"]["columns"],
                    )

            else:
                write = False

        elif what == "DemandCoefficients":
            if self.mode == "sfc-integrated":
                sheets["e"] = pd.DataFrame(
                    data=0,
                    index=self.indeces["e"]["index"],
                    columns=self.indeces["e"]["columns"],
                )
            else:
                write = False

        elif what == "MoneyRates":
            sheets["global"] = pd.DataFrame(
                data=0,
                index=self.indeces["mr"]["index"],
                columns=self.indeces["mr"]["columns"],
            )

        if write:
            dataframe_to_xlsx(path, **sheets)
            log_time(logger, f"ExcelWriter: file {path} created successfully.")

    def __default_values__(
        self, data: [pd.DataFrame], category: str, **kwargs
    ) -> [pd.DataFrame]:

        """
        DESCRIPTION
        =============
        The function is in charge of finding the default values.

        PARAMETERS
        =============
        data     : the data to fill the missing values
        category : defines which kind of information are suppused to give to
                   the function

        kwargs   : info -> in case of sets, info should be passed.
        """

        if category == "sets":

            info = kwargs.get("info")

            assert (
                info is not None
            ), "For sets, we need the info dictionary to be given to the function."

            for item, default in info["defaults"].items():

                missing_items = data.loc[data[item].isna(), item]

                # if any missing item exists
                if len(missing_items):

                    data.loc[data[item].isna(), item] = eval(default)

                    set_name = kwargs.get("name")
                    self.warnings.append(
                        "{} for {}, {} is missed and filled by default values.".format(
                            item,
                            set_name,
                            data.loc[missing_items.index, info["set"]].values.tolist(),
                        )
                    )

            return data

        elif category == "inputs":
            parameter = kwargs.get("parameter")

            assert parameter is not None, "For inputs, we need to specify the parameter"

            """fill the parameters default values"""

    def p(
        self,
        name: str,
        year: int = None,
        lv=slice(
            None
        ),  # eventual sub-level (requested for demand profiles only, sub-indexed by flow)
        r1=slice(None),  # row level 1
        r2=slice(None),  # row level 2
        c1=slice(None),  # col level 1
        c2=slice(None),  # col level 2
        c3=slice(None),  # col level 3
    ):

        """
        DESCRIPTION
        =============
        Mask function for slicing variables/parameters

        """

        # parameter definition
        if name in self.exogenous:
            if isinstance(self.par_exogenous[name], dict):
                if year not in self.ids.all_years:
                    raise AssertionError(
                        "a year within time horizon must be passed as argument"
                    )
                parameter = self.par_exogenous[name][year]
            else:
                if year not in self.ids.all_years + [None]:
                    raise AssertionError(
                        "if a year is passed, it must be within the time horizon"
                    )
                parameter = self.par_exogenous[name]

        elif name in set(self.endogenous) - set(["BV_E"]):
            if isinstance(self.par_endogenous[name], dict):
                if year not in self.ids.all_years:
                    raise AssertionError(
                        "a year within time horizon must be passed as argument"
                    )
                parameter = self.par_endogenous[name][year]
            else:
                if year not in self.ids.all_years + [None]:
                    raise AssertionError(
                        "if a year is passed, it must be within the time horizon"
                    )
                parameter = self.par_endogenous[name]

        elif name == "BV_E":
            if year not in self.ids.all_years + [None]:
                raise AssertionError(
                    "if a year is passed, it must be within the time horizon"
                )
            parameter = self.par_endogenous[name]

        elif name == "I_st":
            if year != None:
                raise AssertionError("year must not be passed as argument")
            parameter = self.par_exogenous[name]

        else:
            raise AssertionError(
                "name of parameter is not valid or not defined in mask function"
            )

        # slicing keys
        slc = {
            "slice(None, None, None)": slice(None),
            "hh": self.ids.time_slices,
            "fh": self.ids.hourly_products,
            "fy": self.ids.yearly_products,
            "sh": self.ids.hourly_sectors,
            "sy": self.ids.yearly_sectors,
            "th": self.ids.hourly_techs,
            "ty": self.ids.yearly_techs,
            "sc": self.ids.consumption_sectors,
            "ts": self.ids.storages,
            "ts+": self.ids.storages_plus,
            "tsn+": self.ids.storages_non_plus,
            "ts+c": self.ids.storage_plus_couple,
            "nts": [
                tech for tech in self.Technologies if tech not in self.ids.storages
            ],
            "tc": self.ids.capacity_techs,
            "tch": self.ids.hourly_capacity_techs,
            "tcy": self.ids.yearly_capacity_techs,
            "tce": self.ids.capacity_techs_equality,
            "tceh": self.ids.hourly_capacity_techs_equality,
            "tcey": self.ids.yearly_capacity_techs_equality,
            "tcr": self.ids.capacity_techs_range,
            "tcrh": self.ids.hourly_capacity_techs_range,
            "tcry": self.ids.yearly_capacity_techs_range,
            "tcd": self.ids.capacity_techs_demand,
            "tcdh": self.ids.hourly_capacity_techs_demand,
            "tcdy": self.ids.yearly_capacity_techs_demand,
        }

        for item in ["t", "s", "r", "f"]:
            for k in list(self.__sets_frames__[_MI[item]].index):
                slc[k] = []
                slc[k].append(k)

        # slicing
        if isinstance(parameter, dict):
            if lv == slice(None):
                raise AssertionError("a sub-level must be specified before slicing")
            else:
                parameter = parameter[lv]

        if name in ["v", "u", "ef", "BV_U"]:
            sliced_parameter = parameter.cloc[(slc[str(r1)], slc[str(r2)]), :].cloc[
                :, (slc[str(c1)], slc[str(c2)])
            ]

        elif name in [
            "wu",
            "bp",
            "bu",
            "cu",
            "tp",
            "st",
            "bv",
            "xy",
            "xh",
            "qy",
            "qh",
            "CU",
            "CU_mr",
            "af_eq",
            "af_min",
            "af_max",
            "soc",
            "dp",
            "BU",
            "BV",
        ]:
            if str(r1) in slc.keys():
                sliced_parameter = parameter.cloc[slc[str(r1)], :].cloc[
                    :, (slc[str(c1)], slc[str(c2)])
                ]
            else:
                sliced_parameter = parameter.cloc[[r1], :].cloc[
                    :, (slc[str(c1)], slc[str(c2)])
                ]

        elif name in ["E", "E_tld", "e", "m", "BV_E"]:
            if year == None:
                sliced_parameter = parameter.cloc[(slc[str(r1)], slc[str(r2)]), :].cloc[
                    :, (slc[str(c1)], slice(None))
                ]
            else:
                sliced_parameter = parameter.cloc[(slc[str(r1)], slc[str(r2)]), :].cloc[
                    :, (slc[str(c1)], [year])
                ]

        elif name in ["E_tld_diag"]:
            if year == None:
                sliced_parameter = parameter.cloc[(slc[str(r1)], slc[str(r2)]), :].cloc[
                    :, (slc[str(c1)], slc[str(c2)], slice(None))
                ]
            else:
                sliced_parameter = parameter.cloc[(slc[str(r1)], slc[str(r2)]), :].cloc[
                    :, (slc[str(c1)], slc[str(c2)], [year])
                ]

        elif name in ["cap_o", "cap_n", "cap_d", "mr"]:
            if str(c2) in slc.keys():
                if year == None:
                    sliced_parameter = parameter.cloc[:, (slc[str(c1)], slc[str(c2)])]
                else:
                    sliced_parameter = parameter.cloc[[year], :].cloc[
                        :, (slc[str(c1)], slc[str(c2)])
                    ]
            else:
                if year == None:
                    sliced_parameter = parameter.cloc[:, (slc[str(c1)], [c2])]
                else:
                    sliced_parameter = parameter.cloc[[year], :].cloc[
                        :, (slc[str(c1)], [c2])
                    ]

        elif name in ["xy_mix", "cap_o_max"]:
            if year == None:
                sliced_parameter = parameter.cloc[
                    :, (slc[str(c1)], slc[str(c2)], slc[str(c3)])
                ]
            else:
                sliced_parameter = parameter.cloc[[year], :].cloc[
                    :, (slc[str(c1)], slc[str(c2)], slc[str(c3)])
                ]

        elif name in ["I_st"]:
            sliced_parameter = parameter.loc[(slc[str(r1)], slc[str(r2)]), :].loc[
                :, (slc[str(c1)], slc[str(c2)])
            ]

        else:
            raise AssertionError("parameter not defined within the slicer function")

        return sliced_parameter

    def _model_generation(self):

        """
        DESCRIPTION
        =============
        The function generates endogenous/exogenous variables, sets objective function and constraints.
        only constraints independent by exogenous data are generated here.
        this way, this function is called just one time, independently by scenario assumptions.

        """

        p = self.p

        # ===============================================================================================================
        # DEFINITION OF SECTOR-TECHNOLOGY IDENTITY MATRIX
        I_st = pd.DataFrame(
            np.zeros(
                (
                    list(self.__matrices__["v"].values())[0].shape[0],
                    list(self.__matrices__["u"].values())[0].shape[1],
                )
            ),
            index=list(self.__matrices__["v"].values())[0].index,
            columns=list(self.__matrices__["u"].values())[0].columns,
        )

        techs_frame = self.__sets_frames__[_MI["t"]]

        for region in self.Regions:
            for sector in self.ids.production_sectors:
                techs = techs_frame.loc[
                    (techs_frame["SECTOR"] == sector)
                    & (~techs_frame["TYPE"].isin(["storage", "storage+"]))
                ].index.tolist()
                I_st.loc[(region, sector), (region, techs)] = 1

        # ===============================================================================================================
        # DEFINITION OF ENDOGENOUS/EXOGENOUS PARAMETERS
        var = {}
        par = {}

        for item in self.endogenous:
            var[item] = {}

            if item not in [
                "cap_o",
                "cap_n",
                "cap_d",
                "BV_U",
                "BV_E",
            ]:

                for y in self.ids.run_years:
                    var[item][y] = cm.Variable(
                        shape=self.__matrices__[item].shape,
                        nonneg=False,
                        index=self.__matrices__[item].index,
                        columns=self.__matrices__[item].columns,
                    )

                for y in self.ids.warm_up_years + self.ids.cool_down_years:
                    if item in ["xh", "qh", "soc"]:
                        pass
                    else:
                        var[item][y] = cm.Variable(
                            shape=self.__matrices__[item].shape,
                            nonneg=False,
                            index=self.__matrices__[item].index,
                            columns=self.__matrices__[item].columns,
                        )

            elif item == "BV_U":
                for y in self.ids.all_years:
                    var[item][y] = {}

                    for flow in self.ids.emission_by_flows:
                        var[item][y][flow] = cm.Variable(
                            shape=list(self.__matrices__[item][flow].shape),
                            nonneg=False,
                            index=self.__matrices__[item][flow].index,
                            columns=self.__matrices__[item][flow].columns,
                        )

            elif item == "BV_E":
                for flow in self.ids.emission_by_flows:
                    var[item][flow] = {}
                    var[item][flow] = cm.Variable(
                        shape=list(self.__matrices__[item][flow].shape),
                        nonneg=False,
                        index=self.__matrices__[item][flow].index,
                        columns=self.__matrices__[item][flow].columns,
                    )

            else:
                var[item] = cm.Variable(
                    shape=self.__matrices__[item].shape,
                    nonneg=False,
                    index=self.__matrices__[item].index,
                    columns=self.__matrices__[item].columns,
                )

        for item in self.exogenous:
            par[item] = {}

            if item in ["xy_mix", "cap_o_max", "E", "E_tld", "E_tld_diag", "e", "mr"]:
                par[item] = cm.Parameter(
                    shape=self.__matrices__[item].shape,
                    index=self.__matrices__[item].index,
                    columns=self.__matrices__[item].columns,
                )
            elif item == "dp":
                for y in self.ids.run_years:
                    par[item][y] = {}

                    for flow in list(self.__matrices__[item].values())[0].keys():
                        par[item][y][flow] = cm.Parameter(
                            shape=list(self.__matrices__[item].values())[0][flow].shape,
                            index=list(self.__matrices__[item].values())[0][flow].index,
                            columns=list(self.__matrices__[item].values())[0][
                                flow
                            ].columns,
                        )
            elif item == "ef":
                for y in self.ids.all_years:
                    par[item][y] = {}

                    for flow in self.ids.emission_by_flows:
                        par[item][y][flow] = cm.Parameter(
                            shape=list(self.__matrices__[item].values())[0][flow].shape,
                            index=list(self.__matrices__[item].values())[0][flow].index,
                            columns=list(self.__matrices__[item].values())[0][
                                flow
                            ].columns,
                        )
            else:
                for y in self.ids.all_years:
                    try:
                        par[item][y] = cm.Parameter(
                            shape=list(self.__matrices__[item].values())[0].shape,
                            index=list(self.__matrices__[item].values())[0].index,
                            columns=list(self.__matrices__[item].values())[0].columns,
                        )
                    except ValueError:
                        if item in ["wu", "bu", "st", "bv"]:
                            pass
                        else:
                            raise

        par["I_st"] = I_st

        self.par_exogenous = par
        self.par_endogenous = var

        # ===============================================================================================================
        # esm PROBLEM - OBJECTIVE FUNCTION

        # total costs
        # obj = sum([cm.rcsum(cm.rcsum(p('CU',y),0),1) for y in self.ids.all_years])

        # total discounted costs (annuities derived by compounding interests and discounting money value)
        obj = sum([cm.rcsum(cm.rcsum(p("CU_mr", y), 0), 1) for y in self.ids.all_years])

        # total production
        # obj = sum([cm.rcsum(p('xy',y),1) for y in self.ids.all_years])

        esm_obj = cp.Minimize(obj)

        # ===============================================================================================================
        # esm PROBLEM - CONSTRAINTS
        esm_eqs = []

        # new, disposed, operative capacities by year always positive or zero
        esm_eqs.append(p("cap_o") >= 0)
        esm_eqs.append(p("cap_n") >= 0)
        esm_eqs.append(p("cap_d") >= 0)

        # dummy co2 constraint
        # esm_eqs.append( cm.rcsum(cm.rcsum(p('BV_U',2040,lv='f.eCO2'),0),1) +
        #                 cm.rcsum(cm.rcsum(p('BV_E',2040,lv='f.eCO2'),0),1) == 0)

        for y in self.ids.all_years:

            # total yearly production by technology always positive
            esm_eqs.append(p("qy", y) >= 0)
            esm_eqs.append(p("xy", y, c2="nts") >= 0)
            esm_eqs.append(p("CU", y) >= 0)
            esm_eqs.append(p("CU_mr", y) >= 0)

            # capacity balances
            if y == self.ids.warm_up_years[0]:
                esm_eqs.append(p("cap_o", y) == p("cap_n", y) - p("cap_d", y))
            else:
                esm_eqs.append(
                    p("cap_o", y) == p("cap_o", y - 1) + p("cap_n", y) - p("cap_d", y)
                )

            # for technologies like housing/transport, the operative capacity must be used
            # NOTE1: may cause infeasibility in case of rapid decrease of demand
            # if self.ids.capacity_techs_demand:
            #     esm_eqs.append( p('xy',y,c2='tcd') == cm.multiply(p('tp',y,r1='t_ca',c2='tcd'), p('cap_o',y,c2='tcd')) )

            # total emissions by flow and by technology
            esm_eqs.append(p("BV", y) == cm.matmul(p("bv", y), cm.diag(p("qy", y))))
            esm_eqs.append(p("BU", y) == cm.matmul(p("bu", y), cm.diag(p("xy", y))))

            for flow in self.ids.emission_by_flows:
                # total emissions, intermediate
                esm_eqs.append(
                    p("BV_U", y, lv=flow)
                    == cm.multiply(
                        cm.multiply(
                            p("ef", y, lv=flow),
                            cm.matmul(p("u", y), cm.diag(p("xy", y))),
                        ),
                        cm.trsp(p("bv", y, r1=flow)),
                    )
                )
                # total emissions, final
                esm_eqs.append(
                    p("BV_E", y, lv=flow)
                    == cm.multiply(p("E", y), cm.trsp(p("bv", y, r1=flow)))
                )

            # cost items
            # names of emissions types are the same added as labels of unit emissions cost
            for item in (
                list(_COST_ITEMS.keys())
                + self.ids.emission_by_flows
                + self.ids.emission_by_techs
            ):

                # fuel and operative costs
                if item in ["c_fu", "c_op"]:
                    esm_eqs.append(
                        p("CU", y, r1=item)
                        == cm.matmul(p("cu", y, r1=item), cm.diag(p("xy", y)))
                    )

                # costs related to emissions (by flow)
                elif item in self.ids.emission_by_flows:
                    esm_eqs.append(
                        p("CU", y, r1=item)
                        == cm.multiply(
                            p("cu", y, r1=item), cm.rcsum(p("BV_U", y, lv=item), 0)
                        )
                    )

                # costs related to emissions (by technology)
                elif item in self.ids.emission_by_techs:
                    esm_eqs.append(
                        p("CU", y, r1=item)
                        == cm.multiply(p("cu", y, r1=item), p("BU", y, r1=item))
                    )

                else:
                    # investment cost
                    esm_eqs.append(
                        p("CU", y, r1="c_in", c2="tc")
                        == cm.multiply(p("cu", y, r1="c_in", c2="tc"), p("cap_n", y))
                    )

                    # decommissioning cost
                    esm_eqs.append(
                        p("CU", y, r1="c_ds", c2="tc")
                        == cm.multiply(p("cu", y, r1="c_ds", c2="tc"), p("cap_d", y))
                    )

        # constraints applied to warm-up and cool-down periods (yearly resolution)
        for y in self.ids.warm_up_years + self.ids.cool_down_years:

            # production of storage technology zero (avoids free energy generation)
            esm_eqs.append(p("xy", y, c2="ts") == 0)

            # production balance, yearly flows
            esm_eqs.append(
                cm.trsp(p("qy", y))
                - cm.matmul(p("u", y), cm.trsp(p("xy", y)))
                - cm.rcsum(p("E", y), 1)
                == 0
            )

            # production balance, yearly sectors
            esm_eqs.append(
                cm.matmul(p("v", y), cm.trsp(p("qy", y)))
                - cp.matmul(p("I_st").values, cm.trsp(p("xy", y)))
                == 0
            )

            # capacity constraints (all capacity techs, averaged yearly availability)
            if self.ids.capacity_techs_equality:
                esm_eqs.append(
                    p("xy", y, c2="tce")
                    == cp.multiply(
                        cm.rcsum(p("af_eq", y, c2="tce"), 0)
                        * (1 / self.ids.period_length),
                        cp.multiply(
                            p("cap_o", y, c2="tce"), p("tp", y, r1="t_ca", c2="tce")
                        ),
                    )
                )

            if self.ids.capacity_techs_range:
                esm_eqs.append(
                    p("xy", y, c2="tcr")
                    >= cp.multiply(
                        cm.rcsum(p("af_min", y, c2="tcr"), 0)
                        * (1 / self.ids.period_length),
                        cp.multiply(
                            p("cap_o", y, c2="tcr"), p("tp", y, r1="t_ca", c2="tcr")
                        ),
                    )
                )
                esm_eqs.append(
                    p("xy", y, c2="tcr")
                    <= cp.multiply(
                        cm.rcsum(p("af_max", y, c2="tcr"), 0)
                        * (1 / self.ids.period_length),
                        cp.multiply(
                            p("cap_o", y, c2="tcr"), p("tp", y, r1="t_ca", c2="tcr")
                        ),
                    )
                )

        # constraints applied to run period (hourly resolution)
        for y in self.ids.run_years:

            # total hourly production always positive (except for storages)
            esm_eqs.append(p("qh", y) >= 0)
            esm_eqs.append(p("xh", y, c2="nts") >= 0)

            # grid to storage only if activated
            # esm_eqs.append( p('xh',y,c2='ts+') <= 0 )

            # for x,q: summing production by hours and nesting into production by year
            esm_eqs.append(
                p("xy", y, c2="th")
                == cm.multiply(
                    cm.rcsum(p("xh", y, c2="th"), 0), (8760 / self.ids.period_length)
                )
            )
            esm_eqs.append(
                p("qy", y, c2="fh")
                == cm.multiply(
                    cm.rcsum(p("qh", y, c2="fh"), 0), (8760 / self.ids.period_length)
                )
            )

            # production balance, yearly flows/sectors
            esm_eqs.append(
                cm.trsp(p("qy", y, c2="fy"))
                - cm.matmul(p("u", y, r2="fy"), cm.trsp(p("xy", y)))
                - cm.rcsum(p("E_tld", y, r2="fy"), 1)
                == 0
            )

            esm_eqs.append(
                cm.matmul(p("v", y, r2="sy"), cm.trsp(p("qy", y)))
                - cp.matmul(p("I_st", r2="sy").values, cm.trsp(p("xy", y)))
                == 0
            )

            # production balance, hourly flows/sectors
            for flow in self.ids.hourly_products:
                esm_eqs.append(
                    cm.trsp(p("qh", y, c2=flow))
                    - cp.matmul(
                        cm.matmul(
                            p("u", y, r2=flow, c2="ty"), cm.diag(p("xy", y, c2="ty"))
                        )
                        * (self.ids.period_length / 8760),
                        cm.trsp(p("dp", y, lv=flow, c2="ty")),
                    )
                    - cm.matmul(
                        p("u", y, r2=flow, c2="th"), cm.trsp(p("xh", y, c2="th"))
                    )
                    - cm.matmul(
                        p("E_tld_diag", y, r2=flow),
                        cm.trsp(p("dp", y, lv=flow, c2="sc")),
                    )
                    == 0
                )

            esm_eqs.append(
                cm.matmul(p("v", y, r2="sh", c2="fh"), cm.trsp(p("qh", y, c2="fh")))
                - cp.matmul(
                    p("I_st", r2="sh", c2="th").values, cm.trsp(p("xh", y, c2="th"))
                )
                == 0
            )

            # availability constraints on hourly technologies
            # WARNING: pay attention to the dimension of availability factors. in case of cars (cap=units), availability should be km/h travelled.
            if self.ids.hourly_capacity_techs_equality:
                esm_eqs.append(
                    p("xh", y, c2="tceh")
                    == cm.multiply(p("af_eq", y, c2="tceh"), p("cap_o", y, c2="tceh"))
                )

            if self.ids.hourly_capacity_techs_range:
                esm_eqs.append(
                    p("xh", y, c2="tcrh")
                    >= cm.multiply(p("af_min", y, c2="tcrh"), p("cap_o", y, c2="tcrh"))
                )
                esm_eqs.append(
                    p("xh", y, c2="tcrh")
                    <= cm.multiply(p("af_max", y, c2="tcrh"), p("cap_o", y, c2="tcrh"))
                )

            # availability constraints on yearly technologies
            if self.ids.yearly_capacity_techs_equality:
                esm_eqs.append(
                    p("xy", y, c2="tcey")
                    == cp.multiply(
                        cm.rcsum(p("af_eq", y, c2="tcey"), 0)
                        * (1 / self.ids.period_length),
                        cp.multiply(
                            p("cap_o", y, c2="tcey"), p("tp", y, r1="t_ca", c2="tcey")
                        ),
                    )
                )

            if self.ids.yearly_capacity_techs_range:
                esm_eqs.append(
                    p("xy", y, c2="tcry")
                    >= cp.multiply(
                        cm.rcsum(p("af_min", y, c2="tcry"), 0)
                        * (1 / self.ids.period_length),
                        cp.multiply(
                            p("cap_o", y, c2="tcry"), p("tp", y, r1="t_ca", c2="tcry")
                        ),
                    )
                )

                esm_eqs.append(
                    p("xy", y, c2="tcry")
                    <= cp.multiply(
                        cm.rcsum(p("af_max", y, c2="tcry"), 0)
                        * (1 / self.ids.period_length),
                        cp.multiply(
                            p("cap_o", y, c2="tcry"), p("tp", y, r1="t_ca", c2="tcry")
                        ),
                    )
                )

            # storage constraints
            if self.ids.storages:
                # storage 1: state of charge by hour
                # standard storages
                esm_eqs.append(
                    p("soc", y, c2="tsn+")
                    == cm.multiply(
                        p("st", y, r1="st_soc_start", c2="tsn+"),
                        p("cap_o", y, c2="tsn+"),
                    )
                    + cp.matmul(
                        np.tril(
                            np.ones([self.ids.period_length, self.ids.period_length])
                        ),
                        -p("xh", y, c2="tsn+"),
                    )
                )
                # storage+
                esm_eqs.append(
                    p("soc", y, c2="ts+")
                    == cm.multiply(
                        p("st", y, r1="st_soc_start", c2="ts+"), p("cap_o", y, c2="ts+")
                    )
                    + cp.matmul(
                        np.tril(
                            np.ones([self.ids.period_length, self.ids.period_length])
                        ),
                        -p("xh", y, c2="ts+") - p("xh", y, c2="ts+c"),
                    )
                )

                # storage 2: all soc periods greater than minimum
                esm_eqs.append(
                    p("soc", y)
                    >= cm.multiply(p("st", y, r1="st_soc_min"), p("cap_o", y, c2="ts"))
                )  # check dimensions

                # storage 3: soc periods cannot exceed operative capacity
                esm_eqs.append(p("soc", y) <= p("cap_o", y, c2="ts"))

                # storage 4: charge/discharge rates cannot exceed a given rate (absolute value of production split to avoid non-linearity)
                esm_eqs.append(
                    p("xh", y, c2="ts")
                    <= cm.multiply(p("st", y, r1="st_cd_rate"), p("cap_o", y, c2="ts"))
                )
                esm_eqs.append(
                    -p("xh", y, c2="ts")
                    <= cm.multiply(p("st", y, r1="st_cd_rate"), p("cap_o", y, c2="ts"))
                )

                # storage 5: managing soc

                # optA - soc of first timeslice in year y equal to last timeslice of y-1 (to be used in case of many timeslices)
                # if y == self.ids.run_years[0]:
                #     esm_eqs.append( p('soc',y)[[0],:] == cm.multiply(p('st',y,r1='st_soc_start'), p('cap_o',y,c2='ts')) )
                # else:
                #     esm_eqs.append( p('soc',y)[[0],:] == p('soc',y-1)[[-1],:] )

                # optB - defined by datasheet
                # esm_eqs.append( p('soc',y,r1=self.ids.period_length) == cm.multiply(p('st',y,r1='st_soc_end'), p('cap_o',y,c2='ts')) )

                # optC - no net positive/negative production (soc start == soc end for all the periods)
                esm_eqs.append(cm.rcsum(p("xh", y, c2="tsn+"), 0) == 0)
                esm_eqs.append(
                    cm.rcsum(p("xh", y, c2="ts+"), 0)
                    + cm.rcsum(p("xh", y, c2="ts+c"), 0)
                    == 0
                )
                # esm_eqs.append( p('xy',y,c2='ts+') + p('xy',y,c2='ts+c') == 0 )
                # esm_eqs.append( p('soc',y,c2='ts+')[[0]] == p('soc',y,c2='ts+')[[-1]] )

        self.obj_funct = esm_obj
        self.constraints = esm_eqs

    def _model_completion(self):

        """
        DESCRIPTION
        =============
        this adds constraints that are function of input-data to the list of problem constraints.
        constraints are mostly policy-related.

        """

        # ===============================================================================================================
        # WEIBULL DISTRIBUTION FUNCTION
        # should be improved with warnings if cycling non-capacity techs

        def weib(
            reg: str,
            tech: str,
            rnd: int = 2,  # rounding factor
        ):

            dis_matr = np.zeros([len(self.ids.all_years), len(self.ids.all_years)])
            years = np.arange(1, len(self.ids.all_years) + 1)

            for y, cluster in self._year_cluster("all"):
                tech_life = (
                    self.data["tp"][cluster["tp"]].loc["t_tl", (reg, tech)].copy()
                )
                shape = self.data["tp"][cluster["tp"]].loc["t_ds", (reg, tech)].copy()

                weib_distr = (
                    (shape / tech_life)
                    * (years / tech_life) ** (shape - 1)
                    * np.exp(-((years / tech_life) ** shape))
                )
                weib_distr = np.round(weib_distr, rnd)

                if sum(weib_distr) - 1 > 0.1:
                    print(
                        f"Integral of Weibull distribution function: {sum(weib_distr)} "
                    )
                    print(f"check: {tech} in {reg}")

                dis_matr[:, self.ids.all_years.index(y)] = np.roll(
                    weib_distr, self.ids.all_years.index(y)
                )

            return dis_matr * np.tril(
                np.ones([len(self.ids.all_years), len(self.ids.all_years)])
            )

        cap_dis_weib = {}

        for reg in self.Regions:
            cap_dis_weib[reg] = {}

            for tech in self.ids.capacity_techs:
                cap_dis_weib[reg][tech] = weib(reg, tech)

        # ===============================================================================================================
        # PRESENT VALUE TO ANNUITY FUNCTION (compounding interest)
        def pta(
            reg: str,
            tech: str,
        ):

            pta_matr = np.zeros([len(self.ids.all_years), len(self.ids.all_years)])

            for y_depl, cluster in self._year_cluster("all"):
                econ_life = (
                    self.data["tp"][cluster["tp"]].loc["t_el", (reg, tech)].copy()
                )
                rate_int = self.data["mr"].loc[y_depl, (reg, "mr_i")].copy()

                for y_vint in range(y_depl, self.ids.all_years[-1] + 1):
                    if y_vint - y_depl < econ_life:
                        if rate_int == 0:
                            pta_matr[
                                self.ids.all_years.index(y_vint),
                                self.ids.all_years.index(y_depl),
                            ] = (
                                1 / econ_life
                            )
                        else:
                            pta_matr[
                                self.ids.all_years.index(y_vint),
                                self.ids.all_years.index(y_depl),
                            ] = (rate_int * (1 + rate_int) ** econ_life) / (
                                (1 + rate_int) ** econ_life - 1
                            )

            return pta_matr

        # ===============================================================================================================
        # ADDING DATA-RELATED CONSTRAINTS
        p = self.p

        for reg in self.Regions:

            # discount rate of money
            d_rate = np.array(
                [
                    (
                        1
                        / (1 + self.data["mr"].loc[:, (reg, "mr_d")])
                        ** (np.arange(len(self.ids.all_years)))
                    ).values
                ]
            ).T

            # fuel, variable, disposal, other costs: discounting money
            for cost_item in (
                list(_COST_ITEMS.keys())
                + self.ids.emission_by_flows
                + self.ids.emission_by_techs
            ):

                if cost_item == "c_in":
                    continue
                else:
                    CU_stack = cp.vstack(
                        [p("CU", y, r1=cost_item, c1=reg) for y in self.ids.all_years]
                    )

                    self.constraints.append(
                        cp.vstack(
                            [
                                p("CU_mr", y, r1=cost_item, c1=reg)
                                for y in self.ids.all_years
                            ]
                        )
                        == cp.multiply(CU_stack, d_rate)
                    )

            for tech in self.ids.capacity_techs:

                # disposed capacity
                self.constraints.append(
                    p("cap_d", c1=reg, c2=tech)
                    == cp.matmul(cap_dis_weib[reg][tech], p("cap_n", c1=reg, c2=tech))
                )

                # investment cost: compounding interest, discounting money, annuities calculation
                CU_in_a = cp.matmul(
                    pta(reg, tech),
                    cm.multiply(
                        p("cap_n", c1=reg, c2=tech),
                        cp.vstack(
                            [
                                p("cu", y, r1="c_in", c1=reg, c2=tech)
                                for y in self.ids.all_years
                            ]
                        ),
                    ),
                )

                self.constraints.append(
                    cp.vstack(
                        [
                            p("CU_mr", y, r1="c_in", c1=reg, c2=tech)
                            for y in self.ids.all_years
                        ]
                    )
                    == cp.multiply(CU_in_a, d_rate)
                )

        for y in self.ids.all_years:

            # constraints on generation mix
            xy_mix = self.data["xy_mix"].loc[
                [y], np.isnan(self.data["xy_mix"].loc[y]) == False
            ]

            if xy_mix.empty:
                continue

            xy_mix_cols = xy_mix.columns.droplevel(1)
            xy_sect_prod = cm.matmul(
                cm.matmul(p("xy", y), cm.trsp(self.par_exogenous["I_st"])),
                self.par_exogenous["I_st"],
            ).cloc[:, xy_mix_cols]
            self.constraints.append(
                p("xy", y).cloc[:, xy_mix_cols]
                >= cp.multiply(xy_mix.values, xy_sect_prod)
            )

            # constraints on operative capacity (may be written out of years loop)
            cap_o_max = self.data["cap_o_max"].loc[
                [y], np.isnan(self.data["cap_o_max"].loc[y]) == False
            ]

            if cap_o_max.empty:
                continue

            cap_o_max_cols = cap_o_max.columns.droplevel(1)
            self.constraints.append(p("cap_o", y).cloc[:, cap_o_max_cols] <= cap_o_max)

    def _data_assigment(self):

        """
        DESCRIPTION
        =============
        assign data to exogenous cvxpy parameters

        """

        # PARAMETERS VALUES ASSIGNEMENT

        for item in self.exogenous:

            # non-clustered parameters
            if item in ["E", "e", "mr"]:
                self.par_exogenous[item].value = self.data[item].values.copy()

            elif item in ["xy_mix", "cap_o_max"]:
                values = self.data[item].values.copy()
                values[np.isnan(self.data[item])] = 0
                self.par_exogenous[item].value = values

            elif item in ["E_tld", "E_tld_diag"]:
                E_tld = pd.DataFrame(
                    self.par_exogenous["E"].value.copy(),
                    self.par_exogenous["E"].index,
                    self.par_exogenous["E"].columns,
                )
                E_tld.loc[(slice(None), self.ids.hourly_products), :] = (
                    E_tld.loc[(slice(None), self.ids.hourly_products), :]
                    * self.ids.period_length
                    / 8760
                )
                self.par_exogenous["E_tld"].value = E_tld.values.copy()

                # new reshaped final demand tilde for writing hourly flow balances
                E_tld_diag = pd.DataFrame(
                    0,
                    self.par_exogenous["E_tld_diag"].index,
                    self.par_exogenous["E_tld_diag"].columns,
                )

                for region in self.Regions:
                    E_tld_diag.loc[
                        (region, slice(None)), (slice(None), region, slice(None))
                    ] = E_tld.loc[(region, slice(None)), :].values
                self.par_exogenous["E_tld_diag"].value = E_tld_diag.values.copy()

            # clustered parameters
            else:
                for y, cluster in self._year_cluster("all"):

                    if item == "dp":
                        if y in self.ids.run_years:
                            for flow in self.ids.hourly_products:
                                self.par_exogenous[item][y][flow].value = self.data[
                                    item
                                ][cluster[item]][flow].values.copy()
                        else:
                            pass

                    elif item == "ef":
                        for flow in self.ids.emission_by_flows:
                            self.par_exogenous[item][y][flow].value = self.data[item][
                                cluster[item]
                            ][flow].values.copy()

                    else:
                        self.par_exogenous[item][y].value = self.data[item][
                            cluster[item]
                        ].values.copy()

    def _model_run(
        self,
        solver=None,
        verbose=True,
    ):

        """
        DESCRIPTION
        =============
        create the problem and launch the solver

        """

        # PROBLEM SOLVING

        self.problem = cp.Problem(self.obj_funct, self.constraints)
        self.problem.solve(solver=solver, verbose=verbose)

        # RESULTS DATAFRAMES
        results = {}

        if self.problem.status == "optimal":
            for var_key, var in self.par_endogenous.items():

                results[var_key] = {}

                if var_key in ["cap_o", "cap_n", "cap_d"]:
                    results[var_key] = cm.cDataFrame(var)

                elif var_key in ["BV_U"]:
                    for year, value in var.items():
                        results[var_key][year] = {}

                        for flow, item in value.items():
                            results[var_key][year][flow] = cm.cDataFrame(item)

                elif var_key in ["BV_E"]:
                    for flow, item in var.items():
                        results[var_key][flow] = cm.cDataFrame(item)

                else:
                    for year, value in var.items():
                        results[var_key][year] = cm.cDataFrame(value)

            # adding some of the par_ex to the results
            for var_key in ["E_tld", "dp"]:
                results[var_key] = {}
                var = self.par_exogenous[var_key]

                if var_key == "dp":
                    for year, fuels in var.items():
                        results[var_key][year] = {}
                        for fuel, values in fuels.items():
                            results[var_key][year][fuel] = cm.cDataFrame(values)

                else:
                    results[var_key] = cm.cDataFrame(var)

        self.results = results

    def _save_results(self, path):

        if not hasattr("results", self):
            raise ValueError("results does not exist")

        dict_to_file(self.results, path, _format="csv")


#%% model test


if __name__ == "__main__":

    from esm import set_log_verbosity

    set_log_verbosity("critical")

    from esm import Model
    from esm.utils import cvxpyModified as cm
    from esm.utils import constants
    import pandas as pd
    import numpy as np
    import cvxpy as cp

    MOD = Model(
        r"esm\1_tests\set_reader\sets_sample.xlsx",
        integrated_model=False,
    )

    MOD.to_excel(path=r"esm\1_tests\set_reader\sets_sample_code.xlsx", item="sets")
    # check.create_input_excels(r'esm\unit_test\input_excels')
    MOD.read_input_excels(r"esm\1_tests\input_scen_1")
    MOD.Base._model_generation()
    MOD.Base._data_assigment()
    MOD.Base._model_completion()
    MOD.Base._model_run(solver="GUROBI")

    # check.generate_clusters_excel(r'esm\unit_test\time_clusters\clusters.xlsx',  )
    # check.read_clusters(r'esm\unit_test\time_clusters\clusters.xlsx',  )
    # check.to_excel('sets', r'esm\unit_test\set_reader\sets_sample_code.xlsx')
