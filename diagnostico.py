import os

import pandas as pd
from dash import Dash, html, dcc, dash_table, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import base64
import datetime
import io
import threading
import time
import shutil #para mover um arquivo para um outro diretório

df = px.data.iris()


print(df.columns)
dataset = pd.read_csv("data.csv")




M = dataset[dataset["diagnosis"]=="M"]
B = dataset[dataset["diagnosis"]=="B"]


fig1 = px.scatter(x=M.radius_mean, y=M.texture_mean, color=M.diagnosis.map({'M':'Maligno'})).update_traces(marker_color="red")
fig2 = px.scatter(x=B.radius_mean, y=B.texture_mean, color=B.diagnosis.map({'B':'Benigno'})).update_traces(marker_color="blue")
fig3 = go.Figure(data=fig1.data + fig2.data).update_layout(template="plotly_dark",title="Tumor Maligno x Benigno", yaxis={'title':'Texture Mean'}, xaxis={'title':'Radius Mean'})

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG,  'assets/sheet.css'])
app.layout = html.Div([dbc.Card([
           dbc.CardBody([
               html.Br(),
               dbc.Row([dbc.Col([
                      html.Div([
                        dbc.Card(
                            dbc.CardBody([
                            ], ), color='#000000',
                            style={"opacity": 0.5, "height": 600, "width": 670, "border-radius": 0})
                    ],)
               ]),
                  dbc.Col([
                      html.Div([
                        dbc.Card(
                            dbc.CardBody([
                            ], ), color='#000000',
                            style={"opacity": 0.5, "height": 430, "width": 620, "border-radius": 0})
                    ],)
               ])
           ])
    ], style={"height":680})

  ]),
    html.Div([dcc.Graph(id="grafico", figure=fig3.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
             paper_bgcolor='rgba(0, 0, 0, 0)'))], style={"position":"absolute","height":420, "width":620, "top":10, "right":25, "overflowY":"hidden"}),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A(id='link', children=['Select Files'])
            ]),
            style={
                'width': '500px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
  ], style={"position":"absolute", "right":100, "top":550}),
  html.Div([], id='upload', style={"position": "absolute", "height": 600, "width": 670, "top": 50, "left": 25}),

  html.Div([
        html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
    ], id='htmlDownload', style={'position':'absolute', 'top':480, 'right':500, 'display':'none'}),
 html.Div([dcc.RadioItems(id='radios', inline=True)], id='htmlRadios', style={'position':'absolute', 'top':480, 'right':190, 'display':'none'}),
  dcc.Interval(id="intervalo", interval=2000),
 html.Div(id='hidden-div', style={'display':'none'})




])


c=0

@app.callback(Output('hidden-div', 'children'),
              [Input("radios", "value")])
def display_value(value):
    global c
    c = value

    return None

@app.callback(
    Output('radios','options'),
    Input('intervalo','n_intervals')
)
def check_pendriver(n_intervals):
    global c
    print(c)
    if os.system('F:').__bool__()==False:
        print("--1")

        return [{'label': 'Área de trabalho', 'value': 1},
        {'label': 'Download', 'value': 2},
        {'label': 'Pen Drive', 'value': 3}]


    else:
        print("--2")
        return [{'label': 'Área de trabalho', 'value': 1},
        {'label': 'Download', 'value': 2},
        {'label': 'Pen Drive', 'value': 3, 'disabled': True}]




#começo arasta
@app.callback(Output('upload', 'children'),
              Output('htmlDownload','style'),
              Output('htmlRadios', 'style'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'), prevent_initial_call=True)
def update_output(list_of_contents, list_of_names, list_of_dates):
    print(f"{'Teste':^20}")
    if list_of_contents is not None:
            print(list_of_contents)
            content_type, content_string = list_of_contents[0].split(',')

            decoded = base64.b64decode(content_string)

            try:
                if 'csv' in list_of_names:
                    # Assume that the user uploaded a CSV file
                    dataset = pd.read_csv(
                        io.StringIO(decoded.decode('utf-8')))





                elif 'xls' in list_of_names:
                    # Assume that the user uploaded an excel file
                    df = pd.read_excel(io.BytesIO(decoded))
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])


            df = pd.read_csv("data.csv")
            df = df.drop("id", axis=1)
            df = df.drop("Unnamed: 32", axis=1)

            x = df.drop("diagnosis", axis=1)
            y = df.diagnosis.values

            scaler = StandardScaler()
            scaler.fit(x)
            standarX = scaler.transform(
                x)  # standarX recebe os dados da variavel x transformados em formato de distribuição gaussiana, torna todos os std(desvio padrão igual a 1

            df2 = pd.DataFrame(standarX)

            x_train, x_test, y_train, y_test = train_test_split(standarX, y, test_size=0.2, random_state=42)

            modelo = GaussianNB()
            modelo.fit(x_train, y_train)

            print(f"Native Bayes score: {modelo.score(x_test, y_test) * 100:.2f}%")

            previsao = modelo.predict(x_test)
            previsoes = pd.DataFrame({"previsao": previsao})
            rows = []
            table_body = [html.Tbody(rows)]
            table_header = [html.Thead(html.Tr([html.Th('Pacientes'),html.Th("Previsões")]))]

            pacientes = []
            c = 1
            for prev in previsao:
                pacientes.append(f"Paciente {c}")
                rows.append(html.Tr([html.Td(f"Paciente {c}",style={"font-size": 15, "height": 40, 'width':325}), html.Td(prev, style={"font-size": 15, "height": 40, 'width':325})]))
                c+=1

            global arquivo
            arquivo = pd.DataFrame({'Pacientes': pacientes, 'Diagnóstico': previsao})




            return dbc.Table(
                    # using the same table as in the above example
                    table_header + table_body,
                    hover=True,
                    responsive=True,
                    striped=True,
            ), {'position':'absolute', 'top':480, 'right':500, 'display':None}, {'position':'absolute', 'top':480, 'right':190, 'display':None}

#fim arasta



def func1():

    time.sleep(2)

    # mover para a área de trabalho
    t = r'C:\Users\Lucia\Downloads\Pacientes.xlsx'
    global c
    print('Testando: ', c)
    if c==1:
      shutil.move(r'C:\Users\Lucia\Downloads\Pacientes.xlsx', r'C:\Users\Lucia\Desktop') #passando arquivo para a Área de trabalho
      t =  r'C:\Users\Lucia\Desktop\Pacientes.xlsx'

    if c==3:
        shutil.move(r'C:\Users\Lucia\Downloads\Pacientes.xlsx', r'F:')#passando arquivo para o Pen Drive
        t = r'F:\Pacientes.xlsx'

    time.sleep(2)

    writer = pd.ExcelWriter(t, engine='xlsxwriter')
    arquivo.to_excel(writer, sheet_name='Pacientes', index=False, na_rep="NaN")

    for column in arquivo:
        tamanho_coluna = max(arquivo[column].astype(str).map(len).max(), len(column))
        index_coluna = arquivo.columns.get_loc(column)
        writer.sheets['Pacientes'].set_column(index_coluna, index_coluna, tamanho_coluna)

    writer.save()



    #se c não for 1 nem 3, então o arquivo será passado na pasta de donwloads como padrão


ci = 0

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    global ci
    if ci==0:
        ci+=1
        threading.Thread(target=func).start()
        threading.Thread(target=func1).start()

    if ci==1:
        ci=0
        return dcc.send_data_frame(arquivo.to_excel, 'Pacientes.xlsx')





if __name__=="__main__":
    app.run_server(debug=True)
