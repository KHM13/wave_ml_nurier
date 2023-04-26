import warnings
import random
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

warnings.filterwarnings("ignore")


# 이상치 확인용 박스플롯 그래프
def box_plot_graph(df, column, width, height):
    layout = go.Layout(template='plotly_white', width=width, height=height)
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Box(y=df[column], name=column, marker=dict(color="#5380b2", outliercolor="#f06548")))

    result = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return result


# 목표변수와 상관관계 분포 파악 그래프
def scatter_graph(df, column):
    df_temp = df.loc[:, [column, "output"]] if column != "output" else df.loc[:, "output"]
    fig = px.parallel_categories(df_temp, color="output", color_continuous_scale=px.colors.sequential.RdBu)
    result = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return result


# 상관관계 heatmap
def correlation_graph(df):
    df_corr = df.corr()

    heat = go.Heatmap(x=df_corr.columns, y=df_corr.index, z=df_corr.values, colorscale=px.colors.sequential.RdBu)
    layout = go.Layout(template='plotly_white', width=1750, height=900)
    fig = go.Figure(data=[heat], layout=layout)

    result = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return result


def main_graph(df, column):
    if column != "output":
        x = df[column].value_counts().sort_index().keys().tolist()
        y1 = df[column].value_counts().sort_index().tolist()
        df_temp = df[[column, 'output']]
        y2_temp = df_temp[(df_temp['output'] == 1)].value_counts().sort_index().tolist()
        y2 = []

        x_temp = df_temp[(df_temp['output'] == 1)].value_counts().keys().sort_values()
        x_copy = x.copy()
        for idx in x_temp:
            x_copy.remove(idx[0])

        i = 0
        for key in x:
            for temp in x_temp:
                value = temp[0]
                if key == value:
                    y2.append(y2_temp.__getitem__(i))
                    i += 1
                    break
                elif x_copy.__contains__(key):
                    y2.append(0)
                    break

        layout = go.Layout(template='plotly_white', width=750, height=500)
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Bar(x=x, y=y1, name=column, marker_color="#5380b2"))
        fig.add_trace(go.Scatter(x=x, y=y2, name='output', marker_color="#f06548"))

        result = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return result
    else:
        x = df[column].value_counts().sort_index().keys().tolist()
        y1 = df[column].value_counts().sort_index().tolist()
        layout = go.Layout(template='plotly_white', width=750, height=500)
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Bar(x=x, y=y1, name=column, marker_color="#5380b2"))

        result = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return result
