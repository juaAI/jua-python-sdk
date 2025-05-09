from jua.settings.jua_settings import JuaSettings


class JuaClient:
    def __init__(self, settings: JuaSettings = JuaSettings()):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
