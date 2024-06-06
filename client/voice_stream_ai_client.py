import asyncio
import websockets
import json
import pyaudio
import threading
import logging

class VoiceStreamClient:
    def __init__(self, uri, callback, language="english", sample_rate=16000, buffer_size=1024, 
                 chunk_length_seconds=3, chunk_offset_seconds=0.1, enable_logging=True):
        self.uri = uri
        self.callback = callback
        self.language = language
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_length_seconds = chunk_length_seconds
        self.chunk_offset_seconds = chunk_offset_seconds
        self.enable_logging = enable_logging
        self.loop = asyncio.new_event_loop()
        self.ws = None
        self.stop_event = threading.Event()

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.sample_rate,
                                      input=True,
                                      frames_per_buffer=self.buffer_size,
                                      stream_callback=self.pyaudio_callback)

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
                "sampleRate": self.sample_rate,
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
                await asyncio.sleep(1)  # Wait a bit before retrying

    def pyaudio_callback(self, in_data, frame_count, time_info, status):
        asyncio.run_coroutine_threadsafe(self.send_audio(in_data), self.loop)
        return (in_data, pyaudio.paContinue)

    def run(self):
        threading.Thread(target=self.loop.run_forever).start()
        asyncio.run_coroutine_threadsafe(self.connect_and_receive(), self.loop)
        self.stream.start_stream()
        logging.debug("Started listening in the background")

    async def connect_and_receive(self):
        await self.connect()
        await self.receive_response()

    def stop(self):
        self.stop_event.set()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        logging.debug("Stopped listening")

# Example callback function to handle responses
def my_callback(response):
    print(f"Received response: {response}")

# Replace with your actual WebSocket server URL
uri = "ws://127.0.0.1:8765"

if __name__ == "__main__":
    client = VoiceStreamClient(uri, my_callback, enable_logging=True)
    client.run()

# To stop the client gracefully, call client.stop() when needed
