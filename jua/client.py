from jua.settings.jua_settings import JuaSettings


class JuaClient:
    def __init__(self, settings: JuaSettings = JuaSettings()):
        self.settings = settings
        self._weather = None

    @property
    def weather(self):
        if self._weather is None:
            from jua.weather._weather import Weather

            self._weather = Weather(self)
        return self._weather

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
