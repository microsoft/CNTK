/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * module.cxx
 *
 * This file is responsible for the module system.  
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

struct Module {
  ModuleFactory fac;
  char *name;
  Module *next;
   Module(const char *n, ModuleFactory f) {
    fac = f;
    name = new char[strlen(n) + 1];
     strcpy(name, n);
     next = 0;
  } ~Module() {
    delete[]name;
  }
};

static Module *modules = 0;

/* -----------------------------------------------------------------------------
 * void Swig_register_module()
 *
 * Register a module.
 * ----------------------------------------------------------------------------- */

void Swig_register_module(const char *n, ModuleFactory f) {
  Module *m = new Module(n, f);
  m->next = modules;
  modules = m;
}

/* -----------------------------------------------------------------------------
 * Language *Swig_find_module()
 *
 * Given a command line option, locates the factory function.
 * ----------------------------------------------------------------------------- */

ModuleFactory Swig_find_module(const char *name) {
  Module *m = modules;
  while (m) {
    if (strcmp(m->name, name) == 0) {
      return m->fac;
    }
    m = m->next;
  }
  return 0;
}
