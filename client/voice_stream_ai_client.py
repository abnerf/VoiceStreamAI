import asyncio
import websockets
import json
import speech_recognition as sr
import threading
import logging

class VoiceStreamClient:
    def __init__(self, uri, callback, language="english", sample_rate=48000, buffer_size=4096, 
                 chunk_length_seconds=3, chunk_offset_seconds=0.1, enable_logging=True):
        self.uri = uri
        self.callback = callback
        self.enable_logging = enable_logging
        self.loop = asyncio.new_event_loop()
        self.stop_event = threading.Event()
        self.ws = None

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.energy_threshold = 300

        self.microphone = sr.Microphone(sample_rate=16000)

        self.language = language
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_length_seconds = chunk_length_seconds
        self.chunk_offset_seconds = chunk_offset_seconds

        if enable_logging:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.CRITICAL)

    async def connect(self):
        logging.debug("Connecting to WebSocket server...")
        self.ws = await websockets.connect(self.uri)

        config = {
            "type": "config",
            "data": {
                "sampleRate": self.samwple_rate,
                "bufferSize": self.buffer_size,
                "channels": 1,
                "language": self.language,
                "processing_strategy": "silence_at_end_of_chunk",
                "processing_args": {
                    "chunk_length_seconds": self.chunk_length_seconds,
                    "chunk_offset_seconds": self.chunk_offset_seconds
                }
            }
        }

        await self.ws.send(json.dumps(config))
        logging.debug("Configuration sent to the server")

    async def send_audio(self, audio_data):
        if self.ws:
            await self.ws.send(audio_data)
            logging.debug("Audio chunk sent to the server")
        else:
            logging.error("WebSocket connection is not established")

    async def receive_response(self):
        while not self.stop_event.is_set():
            if self.ws:
                try:
                    response = await self.ws.recv()
                    logging.debug(f"Response from server: {response}")
                    self.callback(response)
                except websockets.ConnectionClosed:
                    logging.warning("WebSocket connection closed, reconnecting...")
                    await self.connect()
                except Exception as e:
                    logging.error(f"Error receiving response: {e}")
            else:
                logging.error("WebSocket connection is not established")
                await asyncio.sleep(1) 

    def record_callback(self, recognizer, audio):
        data = audio.get_raw_data()

        logging.debug("Captured audio data, sending to server...")
        asyncio.run_coroutine_threadsafe(self.send_audio(data), self.loop)

    async def connect_and_receive(self):
        await self.connect()
        await self.receive_response()

    def run(self):
        threading.Thread(target=self.loop.run_forever).start()
        asyncio.run_coroutine_threadsafe(self.connect_and_receive(), self.loop)
        self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.record_callback, phrase_time_limit=None)

        logging.debug("Started listening in the background")

    def stop(self):
        self.stop_listening(wait_for_stop=False)
        self.stop_event.set()
        self.loop.call_soon_threadsafe(self.loop.stop)

        logging.debug("Stopped listening")

# Callback function to handle responses
def my_callback(response):
    print(f"Received response: {response}")

uri = "ws://127.0.0.1:8765"

if __name__ == "__main__":
    client = VoiceStreamClient(uri, my_callback, enable_logging=True)
    client.run()
