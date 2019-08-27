import pandas as pd
import numpy as np
from pyecharts.charts import *
import pyecharts.options as opts
import os
from functools import reduce

def plot_one_y(df, title:str):
    df.index = pd.to_datetime(df['date'])
    df.sort_index(inplace=True)
    df = df.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="400px"))
            .add_xaxis(xaxis_data=df.index.strftime('%Y-%m-%d').values.tolist())
            .add_yaxis(
            series_name=title.upper(),
            y_axis=np.round(df['r2'].values, 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .set_global_opts(
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_right='45%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )


def plot_two_y(df, title:str):
    # df.index = pd.to_datetime(df['date'])
    # df.sort_index(inplace=True)
    # df = df.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="400px"))
            .add_xaxis(xaxis_data=df.iloc[:,0])
            .add_yaxis(
            series_name=df.columns[1],
            y_axis=df.values[:,1].tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[2],
            y_axis=df.values[:, 2].tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .set_global_opts(
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_right='25%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )

def plot_multi_y(df, title:str):
    df.index = pd.to_datetime(df['date'])
    df.sort_index(inplace=True)
    df = df.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="400px"))
            .add_xaxis(xaxis_data=df.index.strftime('%Y-%m-%d').values.tolist())
            .add_yaxis(
            series_name=df.columns[0],
            y_axis=np.round(df.values[:,0], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[1],
            y_axis=np.round(df.values[:, 1], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[2],
            y_axis=np.round(df.values[:, 2], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[3],
            y_axis=np.round(df.values[:, 3], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[4],
            y_axis=np.round(df.values[:, 4], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[5],
            y_axis=np.round(df.values[:, 5], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[6],
            y_axis=np.round(df.values[:, 6], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[7],
            y_axis=np.round(df.values[:, 7], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[8],
            y_axis=np.round(df.values[:, 8], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .set_global_opts(
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_right='25%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )


def plot_compare_y(df, df2, title:str):
    df.index = pd.to_datetime(df['date'])
    df.sort_index(inplace=True)
    df = df.drop(['date'], axis=1)
    df2 = df2.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
            .add_xaxis(xaxis_data=df.index.strftime('%Y-%m-%d').values.tolist())
            .add_yaxis(
            series_name=df.columns[0],
            y_axis=np.round(df.values[:,0], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[1],
            y_axis=np.round(df.values[:, 1], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[2],
            y_axis=np.round(df.values[:, 2], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[3],
            y_axis=np.round(df.values[:, 3], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[4],
            y_axis=np.round(df.values[:, 4], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[5],
            y_axis=np.round(df.values[:, 5], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[6],
            y_axis=np.round(df.values[:, 6], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[7],
            y_axis=np.round(df.values[:, 7], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[0],
            y_axis=np.round(df2.values[:, 0], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[1],
            y_axis=np.round(df2.values[:, 1], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[2],
            y_axis=np.round(df2.values[:, 2], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[3],
            y_axis=np.round(df2.values[:, 3], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[4],
            y_axis=np.round(df2.values[:, 4], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[5],
            y_axis=np.round(df2.values[:, 5], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[6],
            y_axis=np.round(df2.values[:, 6], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[7],
            y_axis=np.round(df2.values[:, 7], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_left='10%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )

def plot_42_y(df, title:str):
    # df.index = pd.to_datetime(df['date'])
    # df.sort_index(inplace=True)
    # df = df.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1400px", height="1000px"))
            .add_xaxis(xaxis_data=df.values[:,0].astype(int).tolist())
        #     .add_yaxis(
        #     series_name=df.columns[0],
        #     y_axis=np.round(df.values[:,0], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
            .add_yaxis(
            series_name=df.columns[1],
            y_axis=np.round(df.values[:, 1], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[2],
            y_axis=np.round(df.values[:, 2], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[3],
            y_axis=np.round(df.values[:, 3], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[4],
            y_axis=np.round(df.values[:, 4], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[5],
            y_axis=np.round(df.values[:, 5], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[6],
            y_axis=np.round(df.values[:, 6], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[7],
            y_axis=np.round(df.values[:, 7], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[8],
            y_axis=np.round(df.values[:, 8], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[9],
            y_axis=np.round(df.values[:, 9], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[10],
            y_axis=np.round(df.values[:, 10], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[11],
            y_axis=np.round(df.values[:, 11], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[12],
            y_axis=np.round(df.values[:, 12], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[13],
            y_axis=np.round(df.values[:, 13], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[14],
            y_axis=np.round(df.values[:, 14], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[15],
            y_axis=np.round(df.values[:, 15], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[16],
            y_axis=np.round(df.values[:, 16], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[17],
            y_axis=np.round(df.values[:, 17], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[18],
            y_axis=np.round(df.values[:, 18], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[19],
            y_axis=np.round(df.values[:, 19], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[20],
            y_axis=np.round(df.values[:, 20], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[21],
            y_axis=np.round(df.values[:, 21], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[22],
            y_axis=np.round(df.values[:, 22], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[23],
            y_axis=np.round(df.values[:, 23], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[24],
            y_axis=np.round(df.values[:, 24], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[25],
            y_axis=np.round(df.values[:, 25], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[26],
            y_axis=np.round(df.values[:, 26], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[27],
            y_axis=np.round(df.values[:, 27], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[28],
            y_axis=np.round(df.values[:, 28], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[29],
            y_axis=np.round(df.values[:, 29], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[30],
            y_axis=np.round(df.values[:, 30], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[31],
            y_axis=np.round(df.values[:, 31], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[32],
            y_axis=np.round(df.values[:, 32], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[33],
            y_axis=np.round(df.values[:, 33], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[34],
            y_axis=np.round(df.values[:, 34], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[35],
            y_axis=np.round(df.values[:, 35], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
        #     .add_yaxis(
        #     series_name=df.columns[36],
        #     y_axis=np.round(df.values[:, 36], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[37],
        #     y_axis=np.round(df.values[:, 37], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[38],
        #     y_axis=np.round(df.values[:, 38], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[39],
        #     y_axis=np.round(df.values[:, 39], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[40],
        #     y_axis=np.round(df.values[:, 40], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[41],
        #     y_axis=np.round(df.values[:, 41], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[42],
        #     y_axis=np.round(df.values[:, 42], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[43],
        #     y_axis=np.round(df.values[:, 43], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[44],
        #     y_axis=np.round(df.values[:, 44], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[45],
        #     y_axis=np.round(df.values[:, 45], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[46],
        #     y_axis=np.round(df.values[:, 46], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
        #     .add_yaxis(
        #     series_name=df.columns[47],
        #     y_axis=np.round(df.values[:, 47], 2).tolist(),
        #     is_smooth=True,
        #     label_opts=opts.LabelOpts(is_show=False),
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        #     markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        # )
            .set_global_opts(
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_left='0%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False, max_interval=5),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )

if __name__ == '__main__':
    # df = pd.read_csv('result/linear_170/linear_reg.csv')
    # ori_reg_plot = plot_one_y(df, 'linear_170 regression r-sqaure of original data')

    # files = os.listdir('result/linear_170/')
    # name = []
    # for file in files:
    #     if 'normed' in file.split('.')[0].split('_'):
    #         curr_df = pd.read_csv('result/linear_170/' + file)
    #         try:
    #             df = pd.concat([df, curr_df['r2']], axis=1)
    #             name.append(file.split('.')[0].split('_')[-2]+'-w norm')
    #         except:
    #             df = curr_df[['date', 'r2']]
    #             name.append('date')
    #             name.append(file.split('.')[0].split('_')[-2]+'-w/ norm')
    # df.columns = name
    # sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    # sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    # name = ['date'] + [item[0] for item in sort_dict]
    # df = df[name]
    # normed_pca_plot = plot_multi_y(df, 'linear_170 regression r-sqaure after normed pca')
    #
    # files = os.listdir('result/linear_170/')
    # name = []
    # for file in files:
    #     if 'normed' in file.split('.')[0].split('_'):
    #         curr_df = pd.read_csv('result/linear_170/' + file)
    #         try:
    #             df_pca = pd.concat([df_pca, curr_df['pca_ratio']], axis=1)
    #             name.append(file.split('.')[0].split('_')[-2]+'-w norm')
    #         except:
    #             df_pca = curr_df[['date', 'pca_ratio']]
    #             name.append('date')
    #             name.append(file.split('.')[0].split('_')[-2]+'-w/ norm')
    # df_pca.columns = name
    # sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    # sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    # name = ['date'] + [item[0] for item in sort_dict]
    # df_pca = df_pca[name]
    # pca_plot_normed = plot_multi_y(df_pca, 'normed pca eigenvalue ratio')


    # files = os.listdir('result/linear_170/')
    # name = []
    # for file in files:
    #     if 'without' in file.split('.')[0].split('_'):
    #         curr_df = pd.read_csv('result/linear_170/' + file)
    #         try:
    #             df2 = pd.concat([df2, curr_df['r2']], axis=1)
    #             name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
    #         except:
    #             df2 = curr_df[['date', 'r2']]
    #             name.append('date')
    #             name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
    # df2.columns = name
    # sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    # sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    # name = ['date'] + [item[0] for item in sort_dict]
    # df2 = df2[name]
    # without_norm_pca_plot = plot_multi_y(df2, 'linear_170 regression r-sqaure after pca without norm')
    #
    # files = os.listdir('result/linear_170/')
    # name = []
    # for file in files:
    #     if 'without' in file.split('.')[0].split('_'):
    #         curr_df = pd.read_csv('result/linear_170/' + file)
    #         try:
    #             df_pca2 = pd.concat([df_pca2, curr_df['pca_ratio']], axis=1)
    #             name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
    #         except:
    #             df_pca2 = curr_df[['date', 'pca_ratio']]
    #             name.append('date')
    #             name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
    # df_pca2.columns = name
    # sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    # sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    # name = ['date'] + [item[0] for item in sort_dict]
    # df_pca2 = df_pca2[name]
    # pca_plot_without_norm = plot_multi_y(df_pca2, 'pca without normed eigenvalue ratio')
    #
    # files = os.listdir('result/linear_170/')
    # name = []
    # for file in files:
    #     if 'normed' in file.split('.')[0].split('_'):
    #         curr_df = pd.read_csv('result/linear_170/' + file)
    #         try:
    #             df = pd.concat([df, curr_df['r2']], axis=1)
    #             name.append(file.split('.')[0].split('_')[-2]+'-w norm')
    #         except:
    #             df = curr_df[['date', 'r2']]
    #             name.append('date')
    #             name.append(file.split('.')[0].split('_')[-2]+'-w/ norm')
    # df.columns = name
    # sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    # sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    # name = ['date'] + [item[0] for item in sort_dict]
    # df = df[name]
    # normed_pca_plot = plot_multi_y(df, 'linear_170 regression r-sqaure after normed pca')
    # compare_plot = plot_compare_y(df, df2, 'comparison between pca w/ and w/o norm')
    #
    # Page().add(*[ori_reg_plot, normed_pca_plot, pca_plot_normed, without_norm_pca_plot, pca_plot_without_norm, compare_plot]).render(path='2019-07-13.html')


    '''
    ----------------------------------------------------------------------------------------------------------
    PLOT OF DIFFERENT PERIOD
    ----------------------------------------------------------------------------------------------------------
    '''
    # idx_list = [10, 30, 50, 70, 90, 110, 130, 150, 170]
    # files = os.listdir('result/linear_no_pca')
    # files = [file for file in files if int(file.split('.')[0].split('_')[2]) in idx_list]
    # sort_files = {file:file.split('.')[0].split('_')[2] for file in files}
    # sort_files = sorted(sort_files.items(), key=lambda x: int(x[1]))
    # # sort_files = [file for file, _ in sort_files]
    # for file, training_period in sort_files:
    #     curr_df = pd.read_csv('result/linear_no_pca/'+file, index_col=0)
    #     curr_df.columns = [training_period]
    #     try:
    #         output_50 = pd.concat([output_50, curr_df], axis=1, sort=True)
    #     except:
    #         output_50 = curr_df
    # output_50.dropna(inplace=True)
    # output_50['date'] = output_50.index.values
    # reg_plot_50 = plot_multi_y(output_50, '0050 regression r-sqaure with different training length')
    #
    # idx_list = [10, 30, 50, 70, 90, 110, 130, 150, 170]
    # files = os.listdir('result/linear_no_pca_2330')
    # files = [file for file in files if int(file.split('.')[0].split('_')[2]) in idx_list]
    # sort_files = {file:file.split('.')[0].split('_')[2] for file in files}
    # sort_files = sorted(sort_files.items(), key=lambda x: int(x[1]))
    # # sort_files = [file for file, _ in sort_files]
    # for file, training_period in sort_files:
    #     curr_df = pd.read_csv('result/linear_no_pca_2330/'+file, index_col=0)
    #     curr_df.columns = [training_period]
    #     try:
    #         output_2330 = pd.concat([output_2330, curr_df], axis=1, sort=True)
    #     except:
    #         output_2330 = curr_df
    # output_2330.dropna(inplace=True)
    # output_2330['date'] = output_2330.index.values
    # reg_plot_2330 = plot_multi_y(output_2330, '2330 regression r-sqaure with different training length')

    # del output_2330
    # del output_50

    # files = os.listdir('result/linear_no_pca_2330')
    # sort_files = {file:file.split('.')[0].split('_')[2] for file in files}
    # sort_files = sorted(sort_files.items(), key=lambda x: int(x[1]))[:-5]
    # for file, training_period in sort_files:
    #     curr_df = pd.read_csv('result/linear_no_pca_2330/'+file, index_col=0)
    #     curr_df.columns = [training_period]
    #     try:
    #         output_2330 = pd.concat([output_2330, curr_df], axis=1, sort=True)
    #     except:
    #         output_2330 = curr_df
    # output_2330.dropna(inplace=True)
    # output_2330 = output_2330.round(2)
    # files = os.listdir('result/linear_no_pca')
    # sort_files = {file:file.split('.')[0].split('_')[2] for file in files}
    # sort_files = sorted(sort_files.items(), key=lambda x: int(x[1]))[:-5]
    # for file, training_period in sort_files:
    #     curr_df = pd.read_csv('result/linear_no_pca/'+file, index_col=0)
    #     curr_df.columns = [training_period]
    #     try:
    #         output_50 = pd.concat([output_50, curr_df], axis=1, sort=True)
    #     except:
    #         output_50 = curr_df
    #
    # output_50.dropna(inplace=True)
    # output_50 = output_50.round(2)
    # output_2330_mean = output_2330.mean().round(3)
    # output_50_mean = output_50.mean().round(3)
    #
    # average_df = pd.DataFrame(output_50.columns.values.tolist(), columns=['Training Period'])
    # average_df = pd.concat([average_df, pd.DataFrame(output_50_mean.values.tolist(), columns=['R_Square 0050'])], axis=1)
    # average_df = pd.concat([average_df, pd.DataFrame(output_2330_mean.values.tolist(), columns=['R_Square 2330'])], axis=1)
    # mean_plot = plot_two_y(average_df, 'Average r-square of 0050 and 2330')
    #
    # Page().add(*[mean_plot]).render(path='2019-07-20.html')


    '''
    # ----------------------------------------------------------------------------------------------------------
    # PLOT OF DIFFERENT STOCKS - 2019-07-26
    # ----------------------------------------------------------------------------------------------------------
    # '''
    dirs = os.listdir('result/ridge/')
    dirs = [dir for dir in dirs if dir != '.DS_Store']
    # check_sum = reduce(lambda x,y:x+y, range(5,171,5))
    for dir in dirs:
        ticker = dir.split('_')[-1]
        curr_dir = 'result/ridge/' + dir + '/'
        files = os.listdir(curr_dir)
        files = [file for file in files if file != '.DS_Store']
        sort_files = {file:file.split('.')[0].split('_')[2] for file in files}
        sort_files = sorted(sort_files.items(), key=lambda x: int(x[1]))
        sum_file = reduce(lambda x,y:x+y, [int(training_time) for _, training_time in sort_files])
        # if sum_file != check_sum:
        #     print(f'Skip {ticker}')
        #     continue
        for file, training_period in sort_files:
            curr_df = pd.read_csv(curr_dir+file, index_col=0)
            curr_df.columns = [training_period]
            try:
                output = pd.concat([output, curr_df], axis=1, sort=True)
            except:
                output = curr_df
        output.dropna(inplace=True)
        output = output.round(2)
        output_mean = output.mean().round(3)

        try:
            try:
                average_df = pd.concat(
                    [average_df, pd.DataFrame(output_mean.values.tolist(), columns=['{}'.format(ticker)])], axis=1)
            except NameError:
                average_df = pd.DataFrame(output.columns.values.tolist(), columns=['Training Period'])
                average_df = pd.concat(
                    [average_df, pd.DataFrame(output_mean.values.tolist(), columns=['{}'.format(ticker)])], axis=1)
            del output

        except:
            print(f'Skip {ticker}')
            continue
    average_df.dropna(axis=1, inplace=True)
    average_df = average_df.astype(float)
    average_df.to_csv('ridge_mean.csv', index=False)
    # mean_plot = plot_42_y(average_df, 'Average r-square')
    #
    # Page().add(*[mean_plot]).render(path='2019-07-27.html')

    '''
    ----------------------------------------------------------------------------------------------------------
    PLOT OF DIFFERENT STOCKS - 2019-08-03
    ----------------------------------------------------------------------------------------------------------
    '''
    # dirs = os.listdir('result/Tvalues/')
    # dirs = [dir for dir in dirs if dir != '.DS_Store']
    # # check_sum = reduce(lambda x,y:x+y, range(5,171,5))
    # for dir in dirs:
    #     ticker = dir.split('_')[-1]
    #     curr_dir = 'result/Tvalues/' + dir + '/'
    #     files = os.listdir(curr_dir)
    #     files = [file for file in files if file != '.DS_Store']
    #     sort_files = {file:file.split('.')[0].split('_')[2] for file in files}
    #     sort_files = sorted(sort_files.items(), key=lambda x: int(x[1]))
    #
    #
    #     for file, training_period in sort_files:
    #         curr_df = pd.read_csv(curr_dir+file, index_col=0)
    #         curr_df.columns = [training_period]
    #         try:
    #             output = pd.concat([output, curr_df], axis=1, sort=True)
    #         except:
    #             output = curr_df
    #     output.dropna(inplace=True)
    #     output = output.round(2)
    #     output_mean = output.mean().round(3)
    #
    #     try:
    #         try:
    #             average_df = pd.concat(
    #                 [average_df, pd.DataFrame(output_mean.values.tolist(), columns=['{}'.format(ticker)])], axis=1)
    #         except NameError:
    #             average_df = pd.DataFrame(output.columns.values.tolist(), columns=['Training Period'])
    #             average_df = pd.concat(
    #                 [average_df, pd.DataFrame(output_mean.values.tolist(), columns=['{}'.format(ticker)])], axis=1)
    #         del output
    #
    #     except:
    #         print(f'Skip {ticker}')
    #         continue
    # average_df.dropna(axis=1, inplace=True)
    # average_df = average_df.astype(float)
    # average_df.to_csv('ridge_mean.csv', index=False)
    # mean_plot = plot_42_y(average_df, 'Average r-square')
    #
    # Page().add(*[mean_plot]).render(path='2019-08-03.html')