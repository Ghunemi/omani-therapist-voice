import os
import azure.cognitiveservices.speech as speechsdk

def speak_with_azure(text: str, lang: str = "ar-EG") -> str:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_SERVICE_REGION")

    if not speech_key or not service_region:
        raise ValueError("Missing Azure Speech key or region in environment variables.")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_language = lang
    speech_config.speech_synthesis_voice_name = "ar-EG-SalmaNeural"

    audio_filename = f"/tmp/output_{lang}.wav"
    audio_output = speechsdk.audio.AudioOutputConfig(filename=audio_filename)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return audio_filename
    else:
        raise RuntimeError("Azure TTS failed: " + str(result.reason))
