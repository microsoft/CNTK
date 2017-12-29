from director_alternating import *

id = getBar().id()
if id != idFromGetBar():
    raise RuntimeError, "Got wrong id: " + str(id)
