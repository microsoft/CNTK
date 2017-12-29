var director_alternating = require("director_alternating");

id = director_alternating.getBar().id();
if (id != director_alternating.idFromGetBar())
  throw ("Error, Got wrong id: " + str(id));
