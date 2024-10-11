import asyncio
import websockets
import json
import pyaudio
import threading
import logging
import random


class VoiceStreamClient:
    def __init__(self, uri, callback, language="english",
                 sample_rate=16000, buffer_size=1024,
                 chunk_length_seconds=3, chunk_offset_seconds=0.1,
                 enable_logging=True):
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
                    await self.callback(response)
                except websockets.ConnectionClosed:
                    logging.warning(
                        "WebSocket connection closed, reconnecting...")
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


# Replace with your actual WebSocket server URL
uri = "ws://127.0.0.1:8765"


VERSION = "db30a0b46729370aa0807320fcf79cb88b92f6f384e6eb4a52068930505f7a36"
TALON_WEBSOCKET = "ws://localhost:7419/ws"
TALON_WEBSOCKET_ORIGIN = "http://localhost:7419"


class SpeechClient:
    def __init__(self, max_retry_delay=30, enable_logging=True):
        self.active = False
        self.ws = None
        self.final = ""
        self.voice_stream_client = None
        self.enable_logging = enable_logging
        self.max_retry_delay = max_retry_delay

        if enable_logging:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.CRITICAL)

    async def connect(self):
        extra_headers = {
            "Origin": TALON_WEBSOCKET_ORIGIN
        }

        retry_delay = 0.5

        while True:
            try:
                async with websockets.connect(
                        TALON_WEBSOCKET,
                        extra_headers=extra_headers) as websocket:
                    self.ws = websocket
                    logging.debug("Connected")
                    await self.ws.send(json.dumps({"cmd": "hello",
                                                   "version": VERSION}))
                    retry_delay = 0.5
                    await self.handle_messages()
            except websockets.exceptions.ConnectionClosed:
                logging.info("Disconnected. Attempting to reconnect...")
                await self.backoff_delay(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)
            except Exception as e:
                logging.error(f"Error: {e}")
                await self.backoff_delay(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

    async def backoff_delay(self, delay):
        # Add jitter to avoid thundering herd problem
        jitter = random.uniform(0, 0.1 * delay)
        await asyncio.sleep(delay + jitter)

    async def handle_messages(self):
        async for message in self.ws:
            data = json.loads(message)
            cmd = data.get("cmd")
            if cmd == "start":
                lang = data.get("lang")
                logging.info(f"Received start command. Language: {lang}")
                if not self.active:
                    self.active = True
                    await self.start_speech(lang)
            elif cmd == "stop":
                self.active = False
                await self.stop_speech()
            elif cmd == "reload":
                logging.info("Reload requested")
                return

    async def start_speech(self, lang="english"):
        logging.info("Speech recognition started")
        await self.ws.send(json.dumps({"cmd": "start"}))
        # Initialize and start VoiceStreamClient
        if not self.voice_stream_client:
            self.voice_stream_client = VoiceStreamClient(
                uri,
                self.voice_stream_callback,
                language=lang,
                enable_logging=self.enable_logging
            )
            self.voice_stream_client.run()

    async def stop_speech(self):
        logging.info("Speech recognition stopped")
        await self.ws.send(json.dumps({"cmd": "end"}))
        if self.voice_stream_client:
            self.voice_stream_client.stop()
            self.voice_stream_client = None

    async def voice_stream_callback(self, response):
        # This method will be called by VoiceStreamClient
        # when it receives a result
        try:
            response_data = json.loads(response)
            transcript = response_data.get("text", "")
            results = [
                {
                    "final": True,
                    "transcript": transcript,
                    "confidence": response_data.get("confidence", 0.0)
                }
            ]
            await self.ws.send(json.dumps({"cmd": "phrase",
                                           "results": results}))
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from voice stream: {response}")
        except Exception as e:
            logging.error(f"Error in voice stream callback: {e}")


async def main():
    client = SpeechClient()
    await client.connect()

if __name__ == "__main__":
    asyncio.run(main())
