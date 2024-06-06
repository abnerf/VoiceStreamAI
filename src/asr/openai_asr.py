import os
from openai import OpenAI

from .asr_interface import ASRInterface
from src.audio_utils import save_audio_to_file

language_codes = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "czech": "cs",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish": "fi",
    "faroese": "fo",
    "french": "fr",
    "galician": "gl",
    "gujarati": "gu",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "croatian": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "indonesian": "id",
    "icelandic": "is",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "georgian": "ka",
    "kazakh": "kk",
    "khmer": "km",
    "kannada": "kn",
    "korean": "ko",
    "latin": "la",
    "luxembourgish": "lb",
    "lingala": "ln",
    "lao": "lo",
    "lithuanian": "lt",
    "latvian": "lv",
    "malagasy": "mg",
    "maori": "mi",
    "macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "marathi": "mr",
    "malay": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nepali": "ne",
    "dutch": "nl",
    "norwegian nynorsk": "nn",
    "norwegian": "no",
    "occitan": "oc",
    "punjabi": "pa",
    "polish": "pl",
    "pashto": "ps",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "sindhi": "sd",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "sundanese": "su",
    "swedish": "sv",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "turkmen": "tk",
    "tagalog": "tl",
    "turkish": "tr",
    "tatar": "tt",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yiddish": "yi",
    "yoruba": "yo",
    "chinese": "zh",
    "cantonese": "yue",
}

class OpenAIASR(ASRInterface):
    def __init__(self, **kwargs):
        self.asr_pipeline = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        ).audio.transcriptions

    async def transcribe(self, client):
        print("Transcribing audio")
        file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())
        language = None if client.config['language'] is None else language_codes.get(
            client.config['language'].lower())

        with open(file_path, 'rb') as file:
            transcription = self.asr_pipeline.create(
                model="whisper-1",
                file=file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"]
            ).to_dict()

            language = transcription['language']
            text = transcription['text']

            os.remove(file_path)

            print(f"Transcription: {text}") 
            to_return = {
                "language": language,
                "language_probability": 100,
                "text": text,
            }

            return to_return