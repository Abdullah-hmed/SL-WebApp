from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import datetime
from functools import lru_cache
import threading
from ASLAlphabet import frameInference, load_model, getDevice

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

device = getDevice()
model = load_model(device)

@app.route('/')
def index():
    return render_template('index.html')

@lru_cache(maxsize=32)
def cached_frame_inference(image_bytes):
    
    image = Image.open(BytesIO(image_bytes))
        
    return frameInference(image, model, device)

@socketio.on('message')
def handle_message(data):
    threading.Thread(target=process_image, args=(data,)).start()

def process_image(data):
    
    # Process the received image data
    header, encoded = data.split(',', 1)
    
    # Decode the image data
    image_data = base64.b64decode(encoded)
    
    # # Load the image using PIL
    # image = Image.open(BytesIO(image_data))
    
    # # Process the image
    # pred = frameInference(image)
    
    pred = cached_frame_inference(image_data)
    
    if pred is not None:
        
        pred_class, pred_confidence = pred
        
        response_message = {
            'class': pred_class, 
            'confidence': pred_confidence
        }
        print(response_message)
        socketio.emit('prediction', response_message)
        
        print("Received image data")
    else:
        socketio.emit('prediction', {'class': "No Hand Detected", 'confidence': 0})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3002, debug=True)
