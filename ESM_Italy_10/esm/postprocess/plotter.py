# -*- coding: utf-8 -*-
"""
module contains class plot for post-processing of esm model

@author: Amin
"""


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from copy import deepcopy as dc
import math
from esm.utils.constants import _MI
from plotly.subplots import make_subplots
import plotly.offline as pltly
from esm.utils.tools import dict_to_file

YEARLY_VARS = ["xy", "qy"]
HOURLY_VARS = ["xh", "qh", "soc"]

LEVELS = {
    "scenarios": 0,
    "products": 2,
    "regions": 1,
    "technologies": 2,
    "hourly_techs": 2,
    "emission": 2,
    "cost_items": 2,
    "sector":2,
    "sectors":2,
}


colors = [
    "#051c2c",
    "#00a9f4",
    "#2251ff",
    "#aae6f0",
    "#3c96b4",
    "#8c5ac8",
    "#e6a0c8",
    "#d0d0d0",
]

def _attach_yearly_vars(var_dict):
    output = {}
    for yy, vals in var_dict.items():
        for item, df in vals.items():
            if item not in output:
                output[item] = pd.DataFrame()

            idx = df.index
            new_idx = [
                [yy] * len(df),
                idx.get_level_values(0),
                idx.get_level_values(1),
            ]
            df.index = new_idx
            output[item] = pd.concat([output[item], df])

    return output


def attatch_scenarios(results):
    
    base_scenario = [*results][0]
    output = {}
    for var,value in results[base_scenario].items():
        if isinstance(value,pd.DataFrame):
            output[var] = pd.concat({ss:results[ss][var] for ss in results},axis=1)
            
        else:
            output[var] = {}
            for inner_key,inner_value in value.items():
                output[var][inner_key] = pd.concat({ss:results[ss][var][inner_key] for ss in results},axis=1)
                
                
    return output


def plotter(fig, layout, path):
    fig.update_layout(layout)
    fig.write_html(path)

def aggregator(frame,aggregation,instance):
    names = {}
    colors = {}
    for key, info in aggregation.items():
        for val in info["items"]:

            if val not in frame.columns.unique(-1):
                raise ValueError(val)
            frame = frame.rename(mapper={val: key}, axis=1, level=-1)

        names[key] = key
        colors[key] = info["color"]
        frame = frame.groupby(axis=1, level=[0, 1, 2], sort=False).sum()

    names = {**names, **instance._names}
    colors = {**colors, **instance._colors}

    return frame,names,colors

def _reshape_results(model_results):
    results = {}
    for key, value in model_results.items():
        if (isinstance(value, pd.DataFrame)) or (key in ["BV_U", "BV_E", "dp","U","V"]):
            results[key] = value
        else:
            df = pd.DataFrame()
            for year, frame in value.items():

                if key in YEARLY_VARS:
                    frame.index = [year]
                    df = df.append(frame)

                else:
                    frame.index = [[year] * len(frame), frame.index]
                    df = df.append(frame)

                results[key] = df

    return results


def _plot_grids(nplots, ncols):
    if nplots < ncols:
        ncols = nplots
    nrows = math.ceil(nplots / ncols)
    grid = [(row + 1, col + 1) for row in range(nrows) for col in range(ncols)]

    return grid, nrows, ncols


def _plot_df(
    fig, all_data, grid, kinds, slicer, levels, names, colors,total_line,tot_abs=False,
):

    counter = [0]
    # steps will represents the sliders or menues
    for step_index, step in enumerate(
        all_data[0].columns.unique(level=levels["steps"])
    ):

        legends = set()
        for sub_index, sub_item in enumerate(
            all_data[0].columns.unique(level=levels["sub_items"])
        ):

            for data_number, data in enumerate(all_data):

                if total_line:
                    if tot_abs:
                        total = data.loc[:, eval(slicer)].abs().sum(1)
                    else:
                        total = data.loc[:, eval(slicer)].sum(1)

                    fig.add_trace(
                        go.Scatter(
                            x = total.index,
                            y = total.values,
                            marker_color='black',
                            showlegend=False,
                            name='Total'
                            ),
                            row=grid[sub_index][0],
                            col=grid[sub_index][1],
                        )
                for items, values in data.loc[:, eval(slicer)].iteritems():
                    x = values.index
                    y = values.values
                    opacity = 0.5 if np.any(y < 0) else 1
                    # to take the NAME and COLOR easily
                    _id = items[levels["main_items"]]
                    name = names[_id]
                    color = colors[_id]

                    if kinds[data_number] == "bar":
                        fig.add_trace(
                            go.Bar(
                                name=name,
                                x=x,
                                y=y,
                                legendgroup=name,
                                marker_color=color,
                                opacity=opacity,
                                visible=True if step_index == 0 else False,
                                showlegend=False if name in legends else True,
                            ),
                            row=grid[sub_index][0],
                            col=grid[sub_index][1],
                        )

                    elif kinds[data_number] == "line":
                        fig.add_trace(
                            go.Scatter(
                                name=name,
                                x=x,
                                y=y,
                                mode="lines",
                                legendgroup=name,
                                marker_color=color,
                                visible=True if step_index == 0 else False,
                                showlegend=False if name in legends else True,
                                line=dict(width=1),
                            ),
                            row=grid[sub_index][0],
                            col=grid[sub_index][1],
                        )
                        legends.add(name)

                    elif kinds[data_number] == "area":
                        if np.any(y < 0):
                            y_pos = y.copy()
                            y_pos[y_pos < 0] = 0

                            y_neg = y.copy()
                            y_neg[y_neg > 0] = 0

                            yy = [y_pos, y_neg]
                        else:
                            yy = [y]

                        for y in yy:
                            fig.add_trace(
                                go.Scatter(
                                    name=name,
                                    x=x,
                                    y=y,
                                    mode="lines",
                                    stackgroup="two" if np.any(y < 0) else "one",
                                    legendgroup=name,
                                    marker_color=color,
                                    opacity=opacity,
                                    visible=True if step_index == 0 else False,
                                    showlegend=False if name in legends else True,
                                    line=dict(width=0),
                                ),
                                row=grid[sub_index][0],
                                col=grid[sub_index][1],
                            )
                            legends.add(name)

                    legends.add(name)

        counter.append(len(fig.data))

    steps = []
    for step_index, step in enumerate(
        all_data[0].columns.unique(level=levels["steps"])
    ):
        steps.append(
            dict(
                label=names[step],
                method="update",
                args=[
                    {
                        "visible": [
                            True
                            if counter[step_index] <= i < counter[step_index + 1]
                            else False
                            for i in range(len(fig.data))
                        ]
                    },
                ],
            )
        )

    return fig, steps


class Plots:

    """Energy System Optimization Post-Processing Class

    Notes
    -----


    Attributes
    ----------
    scenarios:
        returns a list of scenarios uploaded to the class
    """

    def __init__(self):

        self._data = {}

    def upload_from_file(self, path, scenario, force_overwrite=False):
        """Reads the results from written data files"""
        if (scenario in self.scenarios) and (not force_overwrite):
            raise ValueError(
                f"{scenario} already exists. To overwrite a scenario"
                f" change `force_overwrite=True`."
            )

    def upload_from_model(self, model, scenario, color, force_overwrite=False):
        """Extract the results from a solved model"""
        if (scenario in self.scenarios) and (not force_overwrite):
            raise ValueError(
                f"{scenario} already exists. To overwrite a scenario"
                f" change `force_overwrite=True`."
            )

        if not hasattr(model.Base, "results"):
            raise AttributeError("passed model does not have results object.")

        # is it the first scenario added to the database?
        if not len(self.scenarios):
            self._frames = model.Base.__sets_frames__
            self._ids = model.Base.ids
            self._extract_names_colors()

            for item in ["Flows", "Regions", "Years", "Technologies", "Sectors"]:
                setattr(self, item, getattr(model.Base, item))

        else:
            self._check_consistency(model.Base.__sets_frames__)

        self._names[scenario] = scenario
        self._colors[scenario] = color

        self._data[scenario] = _reshape_results(dc(model.Base.results))

    def plot_mix(
        self,
        path,
        regions,
        scenarios,
        sectors:str,
        what = 'xy',
        period='run',
        steps='regions',
        main_items = 'sector',
        sub_items = 'scenarios',
        steps_mode = 'sliders',
        ncol=3,
        shared_yaxes=True,
        kind='area',
    ):

        titles = {
            'xy': 'Production',
            'cap_o': 'Operative Capacity',
            'cap_n': "New Capacity",
        }
        for item in ["regions", "sectors", "period"]:
            self._check_inputs(item, eval(item))

        years = self._take_years(period)

        unit = self._check_units(sectors, "s", "f")
        techs = self._ids.sectors_techs[sectors]

        all_dfs = {}
        regional = {}
        for scenario in scenarios:

            for index,region in enumerate(regions):
                take = dc(self._data[scenario][what].loc[years,(region,techs)])
                _sum = take.sum(1).values

                for col in range(take.shape[1]):
                    take.iloc[:,col] = take.iloc[:,col].values/_sum

                if not index:
                    df_reg = take
                else:
                    df_reg = pd.concat([df_reg,take],axis=1)

            all_dfs[scenario] = df_reg

        data = [
            pd.concat(all_dfs,axis=1) * 100
        ]

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )
        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")
        slicer = ",".join(slicers)
        _id = main_items.title()



        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind],
            slicer=slicer,
            levels=levels,
            names=self._names,
            colors=self._colors,
            total_line=False
        )

        layout = dict(
            title_text=f"{titles[what]} mix in {self._names[sectors]}",
            xaxis_title="Year",
            yaxis_title='%',
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

        plotter(fig, layout, path)


    def plot_costs_by_items(
        self,
        path,
        regions,
        scenarios,
        cost_items="all",
        to_plot="undiscounted",
        period="run",
        steps="regions",
        main_items="cost_items",
        sub_items="scenarios",
        steps_mode="sliders",
        ncol=3,
        shared_yaxes=True,
        kind="bar",
        unit="",
        total_line=False,
    ):

        if to_plot == "undiscounted":
            take = "CU"
        elif to_plot == "discounted":
            take = "CU_mr"
        else:
            raise ValueError("valid to_plot items are discounted and undiscounted.")


        if cost_items == "all":
            cost_items = self._data[self.scenarios[0]][take].index.unique(-1)


        for item in ["regions", "cost_items", "period"]:
            self._check_inputs(item, eval(item))

        years = self._take_years(period)

        data = [
            pd.concat(
                {
                    scenario: self._data[scenario][take]
                    .groupby(axis=1, level=[0])
                    .sum()
                    .T.stack(-1)
                    .T.loc[years,(regions,cost_items)]
                    for scenario in scenarios
                },
                axis=1,
            )
        ]

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )
        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")
        slicer = ",".join(slicers)
        _id = main_items.title()



        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind],
            slicer=slicer,
            levels=levels,
            names=self._names,
            colors=self._colors,
            total_line=total_line,
        )

        layout = dict(
            title_text=f"{to_plot.title()} Cost",
            xaxis_title="Year",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        plotter(fig, layout, path)

    def plot_costs_by_techs(
        self,
        path,
        regions,
        scenarios,
        techs,
        to_plot="undiscounted",
        period="run",
        steps="regions",
        main_items="cost_items",
        sub_items="scenarios",
        steps_mode="sliders",
        ncol=3,
        shared_yaxes=True,
        kind="bar",
        unit="",
        total_line=False,
        aggregation = None,
    ):

        if to_plot == "undiscounted":
            take = "CU"
        elif to_plot == "discounted":
            take = "CU_mr"
        else:
            raise ValueError("valid to_plot items are discounted and undiscounted.")



        for item in ["regions", "techs", "period"]:
            self._check_inputs(item, eval(item))

        years = self._take_years(period)

        data = [
            pd.concat(
                {
                    scenario: self._data[scenario][take]
                    .groupby(axis=0, level=[0])
                    .sum()
                    .loc[years,(regions,techs)]
                    for scenario in scenarios
                },
                axis=1,
            )
        ]


        if aggregation is not None:
            frame,names,colors = aggregator(data[0],aggregation,self)
            data = [frame]
        else:
            names = self._names
            colors = self._colors



        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )
        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")
        slicer = ",".join(slicers)
        _id = main_items.title()



        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind],
            slicer=slicer,
            levels=levels,
            names=self._names,
            colors=self._colors,
            total_line=total_line,
        )

        layout = dict(
            title_text=f"{to_plot.title()} Cost",
            xaxis_title="Year",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        plotter(fig, layout, path)


    def plot_total_techs_production(
        self,
        path: str,
        regions: list,
        techs: list,
        scenarios: list,
        period: str = "run",
        steps="regions",
        main_items="technologies",
        sub_items="scenarios",
        kind="bar",
        steps_mode="sliders",
        ncol=3,
        shared_yaxes=True,
        total_line=False,
        tot_abs=False,
    ):
        for item in ["techs", "regions", "scenarios", "period"]:
            self._check_inputs(item, eval(item))

        sectors = list({self._ids.techs_sectors[tech] for tech in techs})
        unit = self._check_units(sectors, "s", "f")
        years = self._take_years(period)

        data = [
            pd.concat(
                {
                    scenario: self._data[scenario]["xy"].loc[years, (regions, techs)]
                    for scenario in scenarios
                },
                axis=1,
            )
        ]

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )
        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")
        slicer = ",".join(slicers)
        _id = main_items.title()

        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind],
            slicer=slicer,
            levels=levels,
            names=self._names,
            colors=self._colors,
            total_line=total_line,
            tot_abs=tot_abs,
        )

        layout = dict(
            title_text="Yearly Production",
            xaxis_title="Year",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        plotter(fig, layout, path)

    def plot_sector_capacity(
        self,
        path: str,
        regions: list,
        sectors: list,
        scenarios: list,
        period: str = "run",
        steps="regions",
        main_items="technologies",
        sub_items="scenarios",
        kind="bar",
        steps_mode="sliders",
        ncol=3,
        to_show=["cap_n", "cap_d"],
        shared_yaxes=True,
        total_line=False,
    ):

        for item in ["sectors", "regions", "scenarios", "period"]:
            self._check_inputs(item, eval(item))

        unit = self._check_units(sectors, "s", "s_c")
        years = self._take_years(period)

        technologies = []
        for sector in sectors:
            for tech in self._ids.sectors_techs[sector]:
                if tech in self._ids.capacity_techs:
                    technologies.append(tech)

        data = []
        kinds = []

        if isinstance(to_show,str):
            to_show = [to_show]

        if "cap_n" in to_show:
            data.append(
                pd.concat(
                    {
                        scenario: self._data[scenario]["cap_n"].loc[
                            years, (regions, technologies)
                        ]
                        for scenario in scenarios
                    },
                    axis=1,
                )
            )
            kinds.append(kind)

        if "cap_d" in to_show:

            data.append(
                -pd.concat(
                    {
                        scenario: self._data[scenario]["cap_d"].loc[
                            years, (regions, technologies)
                        ]
                        for scenario in scenarios
                    },
                    axis=1,
                )
            )
            kinds.append(kind)

        if "cap_o" in to_show:
            data.append(
                pd.concat(
                    {
                        scenario: self._data[scenario]["cap_o"].loc[
                            years, (regions, technologies)
                        ]
                        for scenario in scenarios
                    },
                    axis=1,
                )
            )

            if "cap_n" in to_show:
                kinds.append("line")
            else:
                kinds.append(kind)

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )
        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")
        slicer = ",".join(slicers)
        _id = main_items.title()

        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=kinds,
            slicer=slicer,
            levels=levels,
            names=self._names,
            colors=self._colors,
            total_line=total_line,
        )

        type_titles = {
            'cap_o': "Operative",
            'cap_d': "Dicommissioned",
            'cap_n': "New Installed",
            }
        layout = dict(
            title_text="Capacity: "+", ".join([type_titles[ii] for ii in to_show]),
            xaxis_title="Year",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        plotter(fig, layout, path)

    def plot_hourly_flows(
        self,
        regions: list,
        hourly_flows: list,
        scenarios: list,
        year: str,
        steps="scenarios",
        main_items="hourly_flows",
        sub_items="region",
        kind="bar",
        step_mode="sliders",
        ncol=3,
    ):
        pass

    def plot_use(
        self,
        regions,
        flow,
        scenarios,
        path,
        period='run',
        steps="regions",
        main_items="sectors",
        sub_items="scenarios",
        steps_mode="sliders",
        kind="bar",
        ncol=3,
        shared_yaxes=True,
        total_line=False,
        aggregation = None,
    ):
        for item in ["flow", "regions", "scenarios", "period"]:
            self._check_inputs(item, eval(item))

        unit = self._check_units(flow, "f", "f")
        years = self._take_years(period)

        U = pd.DataFrame(
            index=years,
            columns = pd.MultiIndex.from_product(
                [scenarios,regions,self._ids.consumption_sectors+self.Technologies])
            )

        # Filling up the intermediate demand
        sectors = self.Technologies + self._ids.consumption_sectors
        for scenario in scenarios:
            for reg in regions:
                for yy in years:
                    cols = (scenario,reg,self.Technologies)
                    U.loc[yy,cols] = self._data[scenario]['U'][yy].loc[(reg,flow),(reg,self.Technologies)].values

                for sector in self._ids.consumption_sectors:
                    cols = (scenario,reg,sector)
                    U.loc[years,cols] = self._data[scenario]['E'].loc[(reg,flow),(sector,years)].values


        if aggregation is not None:

            frame,names,colors = aggregator(U,aggregation,self)

        else:
            names = self._names
            colors = self._colors

        data = [frame]

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )
        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")
        slicer = ",".join(slicers)
        _id = main_items.title()

        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind],
            slicer=slicer,
            levels=levels,
            names= names,
            colors= colors,
            total_line=total_line,
            tot_abs=False,
        )

        layout = dict(
            title_text=f"{self._names[flow]} Consumption",
            xaxis_title="Year",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        plotter(fig, layout, path)

    def plot_hourly_techs(
        self,
        path: str,
        regions: list,
        hourly_techs: list,
        scenarios: list,
        year: str,
        hourly_flow: str = None,
        steps="regions",
        main_items="hourly_techs",
        sub_items="scenarios",
        steps_mode="sliders",
        kind="bar",
        ncol=3,
        shared_yaxes=True,
        total_line=False,
    ):

        items = ["hourly_techs", "regions", "scenarios", "year"]
        if hourly_flow is not None:
            items.append("hourly_flow")

        for item in items:
            self._check_inputs(item, eval(item))

        # finding sectors
        sectors = list(set([self._ids.techs_sectors[tech] for tech in hourly_techs]))
        unit = self._check_units(sectors, "s", "f")

        data = [
            pd.concat(
                {
                    scenario: self._data[scenario]["xh"].loc[
                        year, (regions, hourly_techs)
                    ]
                    for scenario in scenarios
                },
                axis=1,
            )
        ]

        if main_items == "hourly_techs":
            _id = "Technologies"
        else:
            _id = main_items.title()

        names = self._names
        colors = self._colors

        if hourly_flow:
            names["Final demand"] = "Final demand"
            colors["Final demand"] = "black"
            for i, scenario in enumerate(scenarios):
                E_tld = self._data[scenario]["E"].loc[
                    (slice(None), hourly_flow), (slice(None), year)
                ]

                dp = self._data[scenario]["dp"][year][hourly_flow].loc[
                    :, (slice(None), self._ids.consumption_sectors)
                ]

                for col, value in dp.iteritems():
                    if i == 0:
                        flow_data = pd.DataFrame(
                            index=value.index,
                            columns=pd.MultiIndex.from_product(
                                [
                                    scenarios,
                                    regions,
                                    self._ids.consumption_sectors,
                                    ["Final demand"],
                                ]
                            ),
                        )

                    flow_data.loc[
                        value.index, (scenario, col[0], col[1], "Final demand")
                    ] = (E_tld.loc[(col[0], slice(None)), col[1]].values * value.values)

            flow_data = flow_data.groupby(axis=1, level=[0, 1, -1], sort=False).sum()
            data.append(flow_data)

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )

        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")

        slicer = ",".join(slicers)

        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind] + ["line"],
            slicer=slicer,
            levels=levels,
            names=names,
            colors=colors,
            total_line=total_line,
        )

        layout = dict(
            title_text=f"Production by technologies {year}",
            xaxis_title="Hour",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

        fig.add_shape(
            type="line",
            x0=data[0].min(),
            y0=0,
            x1=data[0].max(),
            y1=0,
            line=dict(color="red",),
        )
        plotter(fig, layout, path)

    def emissions_by_flow(
        self,
        path: str,
        regions,
        emission,
        scenarios,
        by="technologies",
        period="run",
        sub_items="scenarios",
        main_items="emission",
        steps="regions",
        kind="bar",
        steps_mode="sliders",
        ncol=3,
        shared_yaxes=True,
        aggregation=None,
        total_line=False,
        cummulative = False,
    ):
        cumm = ""
        for item in ["emission", "regions", "scenarios", "period"]:
            self._check_inputs(item, eval(item))

        unit = self._check_units(emission, "f", "f")
        years = self._take_years(period)

        if isinstance(emission, list):
            emission = emission[0]

        if by in ["technologies", "sectors"]:
            frame = pd.DataFrame(
                index=years,
                columns=pd.MultiIndex.from_product(
                    [
                        scenarios,
                        regions,
                        [*self._ids.techs_sectors] + self._ids.consumption_sectors,
                    ]
                ),
            )
            # filling the techs
            for scenario in scenarios:
                BV_U = self._data[scenario]["BV_U"]
                BV_E = self._data[scenario]["BV_E"][emission].sum(axis=0)
                sectors = BV_E.index.unique(level=0)

                for yy in years:
                    df = BV_U[yy][emission].sum()
                    regions = frame.columns.unique(level=1).tolist()
                    techs = df.index.unique(level=1).tolist()
                    frame.loc[yy, (scenario, regions, techs)] = df.loc[
                        (regions, techs)
                    ].values
                    frame.loc[yy, (scenario, regions, sectors)] = BV_E.loc[
                        (sectors, yy)
                    ].values

            if by == "sectors":

                frame = frame.rename(
                    mapper=self._ids.techs_sectors, axis=1, level=-1,  # errors="raise"
                )
                frame = frame.groupby(axis=1, level=[0, 1, 2], sort=False).sum()

        elif by == "fuels":
            frame = pd.DataFrame(
                index=years,
                columns=pd.MultiIndex.from_product(
                    [scenarios, regions, self._ids.products,]
                ),
            )
            for scenario in scenarios:
                BV_U = self._data[scenario]["BV_U"]
                BV_E = (
                    self._data[scenario]["BV_E"][emission]
                    .groupby(axis=1, level=1)
                    .sum()
                )

                for yy in years:
                    df = BV_U[yy][emission].sum(axis=1)
                    regions = frame.columns.unique(level=1).tolist()
                    fuels = df.index.unique(level=1).tolist()
                    frame.loc[yy, (scenario, regions, fuels)] = df.loc[
                        (regions, fuels)
                    ].values

                    frame.loc[yy, (scenario, regions, fuels)] += BV_E.loc[
                        (regions, fuels), yy
                    ].values

        else:
            raise ValueError("is not acceptable.")

        if aggregation is not None:

            frame,names,colors = aggregator(frame,aggregation,self)

        else:
            names = self._names
            colors = self._colors

        data = [frame]

        if cummulative:
            data = [data[0].cumsum()]
            cumm = "Cummulative"

        if sub_items == "emission":
            technologies = data[0].columns.unique(level=-1).tolist()
            sub_items = "technologies"

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_xaxes=True,
            shared_yaxes=shared_yaxes,
        )

        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")

        slicer = ",".join(slicers)

        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind],
            slicer=slicer,
            levels=levels,
            names=names,
            colors=colors,
            total_line=total_line,
        )
        layout = dict(
            title_text="{} {} Emissions by {}".format(cumm,self._names[emission], by.title()),
            xaxis_title="Year",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        plotter(fig, layout, path)

    def plot_flow_production(
        self,
        path: str,
        regions: list,
        products: list,
        scenarios: list,
        period: str = "run",
        steps="regions",
        main_items="products",
        sub_items="scenarios",
        kind="bar",
        steps_mode="sliders",
        ncol=3,
        shared_yaxes=True,
        total_line=False,
    ):

        for item in ["products", "regions", "scenarios", "period"]:
            self._check_inputs(item, eval(item))

        unit = self._check_units(products, "f", "f")
        years = self._take_years(period)
        data = [
            pd.concat(
                {
                    scenario: self._data[scenario]["qy"].loc[years, (regions, products)]
                    for scenario in scenarios
                },
                axis=1,
            )
        ]

        grid, nrows, ncols = _plot_grids(nplots=len(eval(sub_items)), ncols=ncol)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[self._names[i] for i in eval(sub_items)],
            shared_yaxes=shared_yaxes,
        )

        items = {"steps": steps, "main_items": main_items, "sub_items": sub_items}

        levels = {key: LEVELS[value] for key, value in items.items()}

        slicers = []
        for index in range(3):
            if index == LEVELS[sub_items]:
                slicers.append("sub_item")
            elif index == LEVELS[steps]:
                slicers.append("step")
            else:
                slicers.append("slice(None)")

        slicer = ",".join(slicers)

        if main_items == "products":
            _id = "Flows"
        else:
            _id = main_items.title()

        fig, fig_steps = _plot_df(
            fig=fig,
            all_data=data,
            grid=grid,
            kinds=[kind],
            slicer=slicer,
            levels=levels,
            names=self._names,
            colors=self._colors,
            total_line=total_line,
        )

        layout = dict(
            title_text="Production of Flows by Regions",
            xaxis_title="Year",
            yaxis_title=unit,
            legend=dict(bordercolor="black", borderwidth=1,),
        )

        if steps_mode == "sliders":
            layout[steps_mode] = [
                dict(
                    active=0,
                    currentvalue={"prefix": "{}: ".format(steps[:-1].title())},
                    steps=fig_steps,
                    pad=dict(t=50),
                ),
            ]
        else:
            layout[steps_mode] = [
                dict(active=0, buttons=fig_steps, pad=dict(t=50),),
            ]
        if kind == "bar":
            layout["barmode"] = "relative"
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", mirror=True, tickangle=45
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        plotter(fig, layout, path)

    def _check_units(self, items, _set, kind):
        cols = {
            "f": "PRODUCTION UNIT",
            "s_c": "CAPACITY UNIT",
        }

        if isinstance(items,str):
            items = [items]

        units = self._frames[_MI[_set]].loc[items, cols[kind]].unique()

        if len(units) - 1:
            raise ValueError(f"items with different units cannot be plotted.")

        return units[0]

    def _check_consistency(self, frames):
        """Checks if scenarios added are consistent or not"""

        # check the frames equlity
        for key, value in self._frames.items():
            if key == "_type":
                continue
            check = frames.get(key)

            if check is None:
                raise ValueError(
                    f"{key} not found the data of the passed model/data sets"
                )

            cols = [col for col in check.columns if 'COLOR' not in col]
            if not value.loc[:,cols].equals(check.loc[:,cols]):
                raise ValueError(
                    f"inconsistencies between the scenario data sets found on level {key}"
                )

    def _check_inputs(self, item, given):

        originals = {
            "products": self._ids.products,
            "regions": self.Regions,
            "techs": self.Technologies,
            "scenarios": self.scenarios,
            "period": ["run", "warm_up", "cool_down", "all"]+self._ids.all_years,
            "sectors": self._ids.production_sectors,
            "hourly_techs": self._ids.hourly_techs,
            "year": self._ids.run_years,
            "hourly_flow": self._ids.hourly_products,
            "emission": self._ids.emission_by_flows,
            "cost_items": self._data[self.scenarios[0]]["CU"].index.unique(-1),
            'flow': self._ids.products
        }

        if isinstance(given, (str, int)):
            given = [given]

        original = set(originals[item])
        given = set(given)

        difference = given.difference(original)

        if difference:
            raise ValueError(
                f"Following items are not a valid: \n{difference}."
                f"Valid items are: \n{original}"
            )

    def _extract_names_colors(self):
        """Returns the names"""
        sheets = [
            "Technologies",
            "Regions",
            "Sectors",
            "Flows",
        ]

        for i, sheet in enumerate(sheets):
            if not i:
                names = self._frames[sheet]["NAME"]
                colors = self._frames[sheet]["COLOR"]

            else:
                names = names.append(self._frames[sheet]["NAME"])
                colors = colors.append(self._frames[sheet]["COLOR"])

        self._names = names.to_dict()
        self._colors = colors.to_dict()
        self._names.update(
            {
                "c_fu": "Fuel Cost",
                "c_op": "Operative Cost",
                "c_in": "Investment Cost",
                "c_ds": "Dismantling Cost",
            }
        )
        self._colors.update(
            {"c_fu": "red", "c_op": "green", "c_in": "blue", "c_ds": "yellow",}
        )

    @property
    def scenarios(self):
        """Returns a list of scenarios"""
        return [*self._data]

    def _take_years(self,period):

        if isinstance(period,(str,int)):
            period = [period]

        years = []
        for pp in period:
            if pp in ["run", "warm_up", "cool_down", "all"]:
                years.extend(self._ids[pp + "_years"])
            elif pp in self._ids['all_years']:
                years.append(pp)
            else:
                raise ValueError(
                    "A period should represent a year of modelling time horize or a specified period"
                    f" in the model. {pp} is not a valid input."
                    )

        return sorted(list(set(years)))

    def save_results(self, scenario, path, _format="xlsx",unstack=True):
        
        if isinstance(scenario, str):
            scenario = [scenario]
            
        to_saves = {}
        for ss in scenario:
            
            to_save = dc(self._data[ss])
    
            BV_U = {}
            to_save["BV_U"] = _attach_yearly_vars(to_save["BV_U"])
            to_save["U"] = pd.concat(to_save["U"])
            to_save["V"] = pd.concat(to_save["V"])
            del to_save['dp']
            to_saves[ss] = to_save
        

    
        if unstack:
            to_saves = attatch_scenarios(to_saves)
        
        dict_to_file(to_saves, path, _format=_format,stack=unstack)
        

    def add_custom_result(self,rh,lh,equation,name,color):


        rh_scenario = self._data[rh]
        lh_scenario = self._data[lh]

        def recurse_equation(rh_scenario,lh_scenario,results):

            for key,val in rh_scenario.items():
                if isinstance(val,pd.DataFrame):
                    lh = lh_scenario[key]
                    rh = rh_scenario[key]
                    results[key] = eval(equation)

                else:

                    netesd_result = recurse_equation(
                        rh_scenario[key],
                        lh_scenario[key],
                        {}
                        )
                    results[key] = netesd_result

            return results


        results = recurse_equation(rh_scenario,lh_scenario, {})

        self._data[name] = results
        self._names[name] = name
        self._colors[name] = color




