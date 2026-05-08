import sys
import importlib

# Usar cuando todo se queda en caché para que tome la nueva configuración  del config en caso de agregar nuevas path

# eliminar módulo cacheado
if "src.utils.config" in sys.modules:
    del sys.modules["src.utils.config"]

# volver a importar
import src.utils.config as config

print(dir(config))