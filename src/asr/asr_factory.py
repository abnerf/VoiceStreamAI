from .whisper_asr import WhisperASR
from .faster_whisper_asr import FasterWhisperASR
from .openai_asr import OpenAIASR

class ASRFactory:
    @staticmethod
    def create_asr_pipeline(type, **kwargs):
        if type == "whisper":
            return WhisperASR(**kwargs)
        if type == "faster_whisper":
            return FasterWhisperASR(**kwargs)
        if type == "openai":
            return OpenAIASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR pipeline type: {type}")
