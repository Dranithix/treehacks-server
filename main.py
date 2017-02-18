import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit

app = Flask(__name__)
sio = SocketIO(app)

@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index.html')

@sio.on('connect', namespace='/data')
def connect():
    pass

@sio.on('draw', namespace='/data')
def message(data):
    print("message ", data)
    with open("/Users/Tanay/Desktop/test.jpg", "rb") as file:
        frameData = file.read().encode("base64")
        emit('preview', frameData)

@sio.on('disconnect', namespace='/data')
def disconnect():
    print('disconnect')

if __name__ == '__main__':
    sio.run(app)