/*
** Lua stand-alone interpreter
** See Copyright Notice in lua.h
*/


#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define lua_c

#include "lua.h"

#include "lauxlib.h"
#include "lualib.h"


/*
** generic extra include file
*/
#ifdef LUA_USERCONFIG
#include LUA_USERCONFIG
#endif


/*
** definition of `isatty'
*/
#ifdef _POSIX_C_SOURCE
#include <unistd.h>
#define stdin_is_tty()	isatty(0)
#else
#define stdin_is_tty()	1  /* assume stdin is a tty */
#endif



#ifndef PROMPT
#define PROMPT		"> "
#endif


#ifndef PROMPT2
#define PROMPT2		">> "
#endif

#ifndef PROGNAME
#define PROGNAME	"lua"
#endif

#ifndef lua_userinit
#define lua_userinit(L)		openstdlibs(L)
#endif


#ifndef LUA_EXTRALIBS
#define LUA_EXTRALIBS	/* empty */
#endif


static lua_State *L = NULL;

static const char *progname = PROGNAME;

extern int Example_Init(lua_State* L);

static const luaL_reg lualibs[] = {
  {"base", luaopen_base},
  {"table", luaopen_table},
  {"io", luaopen_io},
  {"string", luaopen_string},
  {"math", luaopen_math},
  {"debug", luaopen_debug},
  {"loadlib", luaopen_loadlib},
  /* add your libraries here */
  {"example", Example_Init},
  LUA_EXTRALIBS
  {NULL, NULL}
};



static void lstop (lua_State *l, lua_Debug *ar) {
  (void)ar;  /* unused arg. */
  lua_sethook(l, NULL, 0, 0);
  luaL_error(l, "interrupted!");
}


static void laction (int i) {
  signal(i, SIG_DFL); /* if another SIGINT happens before lstop,
                              terminate process (default action) */
  lua_sethook(L, lstop, LUA_MASKCALL | LUA_MASKRET | LUA_MASKCOUNT, 1);
}


static void print_usage (void) {
  fprintf(stderr,
  "usage: %s [options] [script [args]].\n"
  "Available options are:\n"
  "  -        execute stdin as a file\n"
  "  -e stat  execute string `stat'\n"
  "  -i       enter interactive mode after executing `script'\n"
  "  -l name  load and run library `name'\n"
  "  -v       show version information\n"
  "  --       stop handling options\n" ,
  progname);
}


static void l_message (const char *pname, const char *msg) {
  if (pname) fprintf(stderr, "%s: ", pname);
  fprintf(stderr, "%s\n", msg);
}


static int report (int status) {
  const char *msg;
  if (status) {
    msg = lua_tostring(L, -1);
    if (msg == NULL) msg = "(error with no message)";
    l_message(progname, msg);
    lua_pop(L, 1);
  }
  return status;
}


static int lcall (int narg, int clear) {
  int status;
  int base = lua_gettop(L) - narg;  /* function index */
  lua_pushliteral(L, "_TRACEBACK");
  lua_rawget(L, LUA_GLOBALSINDEX);  /* get traceback function */
  lua_insert(L, base);  /* put it under chunk and args */
  signal(SIGINT, laction);
  status = lua_pcall(L, narg, (clear ? 0 : LUA_MULTRET), base);
  signal(SIGINT, SIG_DFL);
  lua_remove(L, base);  /* remove traceback function */
  return status;
}


static void print_version (void) {
  l_message(NULL, LUA_VERSION "  " LUA_COPYRIGHT);
}


static void getargs (char *argv[], int n) {
  int i;
  lua_newtable(L);
  for (i=0; argv[i]; i++) {
    lua_pushnumber(L, i - n);
    lua_pushstring(L, argv[i]);
    lua_rawset(L, -3);
  }
  /* arg.n = maximum index in table `arg' */
  lua_pushliteral(L, "n");
  lua_pushnumber(L, i-n-1);
  lua_rawset(L, -3);
}


static int docall (int status) {
  if (status == 0) status = lcall(0, 1);
  return report(status);
}


static int file_input (const char *name) {
  return docall(luaL_loadfile(L, name));
}


static int dostring (const char *s, const char *name) {
  return docall(luaL_loadbuffer(L, s, strlen(s), name));
}


static int load_file (const char *name) {
  lua_pushliteral(L, "require");
  lua_rawget(L, LUA_GLOBALSINDEX);
  if (!lua_isfunction(L, -1)) {  /* no `require' defined? */
    lua_pop(L, 1);
    return file_input(name);
  }
  else {
    lua_pushstring(L, name);
    return report(lcall(1, 1));
  }
}


/*
** this macro can be used by some `history' system to save lines
** read in manual input
*/
#ifndef lua_saveline
#define lua_saveline(L,line)	/* empty */
#endif


/*
** this macro defines a function to show the prompt and reads the
** next line for manual input
*/
#ifndef lua_readline
#define lua_readline(L,prompt)		readline(L,prompt)

/* maximum length of an input line */
#ifndef MAXINPUT
#define MAXINPUT	512
#endif


static int readline (lua_State *l, const char *prompt) {
  static char buffer[MAXINPUT];
  if (prompt) {
    fputs(prompt, stdout);
    fflush(stdout);
  }
  if (fgets(buffer, sizeof(buffer), stdin) == NULL)
    return 0;  /* read fails */
  else {
    lua_pushstring(l, buffer);
    return 1;
  }
}

#endif


static const char *get_prompt (int firstline) {
  const char *p = NULL;
  lua_pushstring(L, firstline ? "_PROMPT" : "_PROMPT2");
  lua_rawget(L, LUA_GLOBALSINDEX);
  p = lua_tostring(L, -1);
  if (p == NULL) p = (firstline ? PROMPT : PROMPT2);
  lua_pop(L, 1);  /* remove global */
  return p;
}


static int incomplete (int status) {
  if (status == LUA_ERRSYNTAX &&
         strstr(lua_tostring(L, -1), "near `<eof>'") != NULL) {
    lua_pop(L, 1);
    return 1;
  }
  else
    return 0;
}


static int load_string (void) {
  int status;
  lua_settop(L, 0);
  if (lua_readline(L, get_prompt(1)) == 0)  /* no input? */
    return -1;
  if (lua_tostring(L, -1)[0] == '=') {  /* line starts with `=' ? */
    lua_pushfstring(L, "return %s", lua_tostring(L, -1)+1);/* `=' -> `return' */
    lua_remove(L, -2);  /* remove original line */
  }
  for (;;) {  /* repeat until gets a complete line */
    status = luaL_loadbuffer(L, lua_tostring(L, 1), lua_strlen(L, 1), "=stdin");
    if (!incomplete(status)) break;  /* cannot try to add lines? */
    if (lua_readline(L, get_prompt(0)) == 0)  /* no more input? */
      return -1;
    lua_concat(L, lua_gettop(L));  /* join lines */
  }
  lua_saveline(L, lua_tostring(L, 1));
  lua_remove(L, 1);  /* remove line */
  return status;
}


static void manual_input (void) {
  int status;
  const char *oldprogname = progname;
  progname = NULL;
  while ((status = load_string()) != -1) {
    if (status == 0) status = lcall(0, 0);
    report(status);
    if (status == 0 && lua_gettop(L) > 0) {  /* any result to print? */
      lua_getglobal(L, "print");
      lua_insert(L, 1);
      if (lua_pcall(L, lua_gettop(L)-1, 0, 0) != 0)
        l_message(progname, lua_pushfstring(L, "error calling `print' (%s)",
                                               lua_tostring(L, -1)));
    }
  }
  lua_settop(L, 0);  /* clear stack */
  fputs("\n", stdout);
  progname = oldprogname;
}


static int handle_argv (char *argv[], int *interactive) {
  if (argv[1] == NULL) {  /* no more arguments? */
    if (stdin_is_tty()) {
      print_version();
      manual_input();
    }
    else
      file_input(NULL);  /* executes stdin as a file */
  }
  else {  /* other arguments; loop over them */
    int i;
    for (i = 1; argv[i] != NULL; i++) {
      if (argv[i][0] != '-') break;  /* not an option? */
      switch (argv[i][1]) {  /* option */
        case '-': {  /* `--' */
          if (argv[i][2] != '\0') {
            print_usage();
            return 1;
          }
          i++;  /* skip this argument */
          goto endloop;  /* stop handling arguments */
        }
        case '\0': {
          file_input(NULL);  /* executes stdin as a file */
          break;
        }
        case 'i': {
          *interactive = 1;
          break;
        }
        case 'v': {
          print_version();
          break;
        }
        case 'e': {
          const char *chunk = argv[i] + 2;
          if (*chunk == '\0') chunk = argv[++i];
          if (chunk == NULL) {
            print_usage();
            return 1;
          }
          if (dostring(chunk, "=<command line>") != 0)
            return 1;
          break;
        }
        case 'l': {
          const char *filename = argv[i] + 2;
          if (*filename == '\0') filename = argv[++i];
          if (filename == NULL) {
            print_usage();
            return 1;
          }
          if (load_file(filename))
            return 1;  /* stop if file fails */
          break;
        }
        case 'c': {
          l_message(progname, "option `-c' is deprecated");
          break;
        }
        case 's': {
          l_message(progname, "option `-s' is deprecated");
          break;
        }
        default: {
          print_usage();
          return 1;
        }
      }
    } endloop:
    if (argv[i] != NULL) {
      const char *filename = argv[i];
      getargs(argv, i);  /* collect arguments */
      lua_setglobal(L, "arg");
      return file_input(filename);  /* stop scanning arguments */
    }
  }
  return 0;
}


static void openstdlibs (lua_State *l) {
  const luaL_reg *lib = lualibs;
  for (; lib->func; lib++) {
    lib->func(l);  /* open library */
    lua_settop(l, 0);  /* discard any results */
  }
}


static int handle_luainit (void) {
  const char *init = getenv("LUA_INIT");
  if (init == NULL) return 0;  /* status OK */
  else if (init[0] == '@')
    return file_input(init+1);
  else
    return dostring(init, "=LUA_INIT");
}


struct Smain {
  int argc;
  char **argv;
  int status;
};


static int pmain (lua_State *l) {
  struct Smain *s = (struct Smain *)lua_touserdata(l, 1);
  int status;
  int interactive = 0;
  if (s->argv[0] && s->argv[0][0]) progname = s->argv[0];
  L = l;
  lua_userinit(l);  /* open libraries */
  status = handle_luainit();
  if (status == 0) {
    status = handle_argv(s->argv, &interactive);
    if (status == 0 && interactive) manual_input();
  }
  s->status = status;
  return 0;
}


int main (int argc, char *argv[]) {
  int status;
  struct Smain s;
  lua_State *l = lua_open();  /* create state */
  if (l == NULL) {
    l_message(argv[0], "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  s.argc = argc;
  s.argv = argv;
  status = lua_cpcall(l, &pmain, &s);
  report(status);
  lua_close(l);
  return (status || s.status) ? EXIT_FAILURE : EXIT_SUCCESS;
}

