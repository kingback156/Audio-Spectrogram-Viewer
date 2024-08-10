from flask import Flask, render_template
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import librosa
from datetime import datetime
import numpy as np
import base64
import io

server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/spectrogram/')

app.layout = html.Div([
    html.H1('Audio Spectrogram Viewer', style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-audio',
        children=html.Button('Upload Audio File', style={'fontSize': 20}),
        style={'textAlign': 'center', 'marginBottom': '20px'}
    ),
    html.Div(id='audio-player', style={'textAlign': 'center'}),
    dcc.Graph(id='spectrogram'),
    html.Button('Move Line and Play Audio', id='move-line-button', style={'fontSize': 20}),
    html.Button(id='hidden-play-button', style={'display': 'none'}), 
    dcc.Interval(
        id='interval-component',
        interval=100,
        n_intervals=0,
        disabled=True
    ),
    dcc.Store(id='line-position', data={'x': 0, 'moving': False}),
    dcc.Store(id='audio-content', data=''),
    dcc.Store(id='start-time', data={'start': None})
], style={'textAlign': 'center'})

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    return base64.b64decode(content_string)

def create_spectrogram_figure(y, sr, duration, line_position):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=S_dB,
        x=np.linspace(0, duration, S_dB.shape[1]),
        y=librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2-1),
        colorscale='Plasma'
    ))

    fig.add_trace(go.Scatter(
        x=[line_position['x'], line_position['x']],
        y=[0, np.max(librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2-1))],
        mode='lines',
        line=dict(color='white', width=4),
        name='Red Line'
    ))

    fig.update_layout(
        title=dict(text='<span style="color:red"> Spectrogram </span> of audio signal', font=dict(weight='bold')),
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        yaxis=dict(range=[0, np.max(librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2-1))])
    )
    
    return fig

@app.callback(
    [Output('audio-player', 'children'),
     Output('spectrogram', 'figure'),
     Output('interval-component', 'disabled'),
     Output('line-position', 'data'),
     Output('interval-component', 'interval'),
     Output('audio-content', 'data'),
     Output('start-time', 'data')],
    [Input('upload-audio', 'contents'),
     Input('move-line-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('line-position', 'data'),
     State('audio-content', 'data'),
     State('start-time', 'data')]
)
def update_output(contents, move_clicks, n_intervals, line_position, audio_content, start_time):
    ctx = dash.callback_context
    if not ctx.triggered:
        return '', go.Figure(), True, line_position, dash.no_update, dash.no_update, start_time

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'upload-audio' and contents:
        data = parse_contents(contents)
        y, sr = librosa.load(io.BytesIO(data), sr=None)
        duration = len(y) / sr

        fig = create_spectrogram_figure(y, sr, duration, line_position)

        audio_src = f'data:audio/wav;base64,{contents.split(",")[1]}'
        audio_player = html.Audio(src=audio_src, controls=True, id='audio-element', style={'width': '90%'})
        return audio_player, fig, True, line_position, dash.no_update, contents, {'start': None}

    elif trigger_id == 'move-line-button' and move_clicks:
        start_time = {'start': datetime.now().timestamp()}
        return dash.no_update, dash.no_update, False, {'x': 0, 'moving': True}, dash.no_update, audio_content, start_time

    elif trigger_id == 'interval-component' and line_position['moving']:
        data = parse_contents(audio_content)
        y, sr = librosa.load(io.BytesIO(data), sr=None)
        duration = len(y) / sr

        current_time = datetime.now().timestamp()
        elapsed_time = current_time - start_time['start']

        new_x = min(elapsed_time, duration)
        if new_x >= duration:
            new_x = duration
            line_position['moving'] = False

        line_position['x'] = new_x

        fig = create_spectrogram_figure(y, sr, duration, line_position)

        return dash.no_update, fig, not line_position['moving'], line_position, dash.no_update, dash.no_update, start_time

    return dash.no_update, dash.no_update, True, line_position, dash.no_update, dash.no_update, start_time

@app.callback(
    Output('hidden-play-button', 'n_clicks'),
    [Input('move-line-button', 'n_clicks')]
)
def play_audio_on_button_click(n_clicks):
    if n_clicks:
        return 1
    return dash.no_update

app.clientside_callback(
    """
    function(n_clicks) {
        var audio = document.getElementById('audio-element');
        if(audio) {
            audio.play();
        }
        return '';
    }
    """,
    Output('hidden-play-button', 'children'),
    [Input('hidden-play-button', 'n_clicks')]
)

@server.route('/')
def index():
    return render_template('index.html')

@server.route('/spectrogram.html')
def render_spectrogram():
    return app.index()

if __name__ == '__main__':
    server.run(debug=True)
