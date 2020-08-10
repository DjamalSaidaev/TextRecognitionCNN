import pyttsx3


class Audio:
    def __init__(self):
        self.tts = pyttsx3.init()
        self.eng_voice = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0"
        self.rus_voice = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_RU-RU_IRINA_11.0"

    def speech(self, text, lang):
        if lang == 'ENG':
            self.tts.setProperty('voice', self.eng_voice)
        elif lang == 'RUS':
            self.tts.setProperty('voice', self.rus_voice)
        self.tts.say(text)
        self.tts.runAndWait()
