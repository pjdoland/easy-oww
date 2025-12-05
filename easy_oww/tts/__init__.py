"""TTS (Text-to-Speech) module for synthetic sample generation"""

from easy_oww.tts.piper import PiperTTS
from easy_oww.tts.voices import VoiceDownloader
from easy_oww.tts.generator import SampleGenerator

__all__ = ['PiperTTS', 'VoiceDownloader', 'SampleGenerator']
