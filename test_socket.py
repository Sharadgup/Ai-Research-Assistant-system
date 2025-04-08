from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__,  template_folder='src/templates')
app.config['SECRET_KEY'] = 'secret!'  # Replace with a real secret in production
socketio = SocketIO(app, async_mode='eventlet')

@app.route('/')
def index():
    return render_template('socket_test.html')

@socketio.on('connect')
def test_connect():
    print("Client connected")
    emit('my_response', {'data': 'Connected'})

@socketio.on('test_message')
def test_message(message):
    print(f"Received: {message}")
    emit('my_response', {'data': message['data']})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5001)
