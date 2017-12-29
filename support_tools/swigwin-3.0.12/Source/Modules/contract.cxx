/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * contract.cxx
 *
 * Support for Wrap by Contract in SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

/* Contract structure.  This holds rules about the different kinds of contract sections
   and their combination rules */

struct contract {
  const char *section;
  const char *combiner;
};
/* Contract rules.  This table defines what contract sections are recognized as well as
   how contracts are to combined via inheritance */

static contract Rules[] = {
  {"require:", "&&"},
  {"ensure:", "||"},
  {NULL, NULL}
};

/* ----------------------------------------------------------------------------
 * class Contracts:
 *
 * This class defines the functions that need to be used in 
 *         "wrap by contract" module.
 * ------------------------------------------------------------------------- */

class Contracts:public Dispatcher {
  String *make_expression(String *s, Node *n);
  void substitute_parms(String *s, ParmList *p, int method);
public:
  Hash *ContractSplit(Node *n);
  int emit_contract(Node *n, int method);
  int cDeclaration(Node *n);
  int constructorDeclaration(Node *n);
  int externDeclaration(Node *n);
  int extendDirective(Node *n);
  int importDirective(Node *n);
  int includeDirective(Node *n);
  int namespaceDeclaration(Node *n);
  int classDeclaration(Node *n);
  virtual int top(Node *n);
};

static int Contract_Mode = 0;	/* contract option */
static int InClass = 0;		/* Parsing C++ or not */
static int InConstructor = 0;
static Node *CurrentClass = 0;

/* Set the contract mode, default is 0 (not open) */
/* Normally set in main.cxx, when get the "-contracts" option */
void Swig_contract_mode_set(int flag) {
  Contract_Mode = flag;
}

/* Get the contract mode */
int Swig_contract_mode_get() {
  return Contract_Mode;
}

/* Apply contracts */
void Swig_contracts(Node *n) {

  Contracts *a = new Contracts;
  a->top(n);
  delete a;
}

/* Split the whole contract into preassertion, postassertion and others */
Hash *Contracts::ContractSplit(Node *n) {

  String *contract = Getattr(n, "feature:contract");
  Hash *result;
  if (!contract)
    return NULL;

  result = NewHash();
  String *current_section = NewString("");
  const char *current_section_name = Rules[0].section;
  List *l = SplitLines(contract);

  Iterator i;
  for (i = First(l); i.item; i = Next(i)) {
    int found = 0;
    if (Strchr(i.item, '{'))
      continue;
    if (Strchr(i.item, '}'))
      continue;
    for (int j = 0; Rules[j].section; j++) {
      if (Strstr(i.item, Rules[j].section)) {
	if (Len(current_section)) {
	  Setattr(result, current_section_name, current_section);
	  current_section = Getattr(result, Rules[j].section);
	  if (!current_section)
	    current_section = NewString("");
	}
	current_section_name = Rules[j].section;
	found = 1;
	break;
      }
    }
    if (!found)
      Append(current_section, i.item);
  }
  if (Len(current_section))
    Setattr(result, current_section_name, current_section);
  return result;
}

/* This function looks in base classes and collects contracts found */
void inherit_contracts(Node *c, Node *n, Hash *contracts, Hash *messages) {

  Node *b, *temp;
  String *name, *type, *local_decl, *base_decl;
  List *bases;
  int found = 0;

  bases = Getattr(c, "bases");
  if (!bases)
    return;

  name = Getattr(n, "name");
  type = Getattr(n, "type");
  local_decl = Getattr(n, "decl");
  if (local_decl) {
    local_decl = SwigType_typedef_resolve_all(local_decl);
  } else {
    return;
  }
  /* Width first search */
  for (int i = 0; i < Len(bases); i++) {
    b = Getitem(bases, i);
    temp = firstChild(b);
    while (temp) {
      base_decl = Getattr(temp, "decl");
      if (base_decl) {
	base_decl = SwigType_typedef_resolve_all(base_decl);
	if ((checkAttribute(temp, "storage", "virtual")) &&
	    (checkAttribute(temp, "name", name)) && (checkAttribute(temp, "type", type)) && (!Strcmp(local_decl, base_decl))) {
	  /* Yes, match found. */
	  Hash *icontracts = Getattr(temp, "contract:rules");
	  Hash *imessages = Getattr(temp, "contract:messages");
	  found = 1;
	  if (icontracts && imessages) {
	    /* Add inherited contracts and messages to the contract rules above */
	    int j = 0;
	    for (j = 0; Rules[j].section; j++) {
	      String *t = Getattr(contracts, Rules[j].section);
	      String *s = Getattr(icontracts, Rules[j].section);
	      if (s) {
		if (t) {
		  Insert(t, 0, "(");
		  Printf(t, ") %s (%s)", Rules[j].combiner, s);
		  String *m = Getattr(messages, Rules[j].section);
		  Printf(m, " %s [%s from %s]", Rules[j].combiner, Getattr(imessages, Rules[j].section), Getattr(b, "name"));
		} else {
		  Setattr(contracts, Rules[j].section, NewString(s));
		  Setattr(messages, Rules[j].section, NewStringf("[%s from %s]", Getattr(imessages, Rules[j].section), Getattr(b, "name")));
		}
	      }
	    }
	  }
	}
	Delete(base_decl);
      }
      temp = nextSibling(temp);
    }
  }
  Delete(local_decl);
  if (!found) {
    for (int j = 0; j < Len(bases); j++) {
      b = Getitem(bases, j);
      inherit_contracts(b, n, contracts, messages);
    }
  }
}

/* This function cleans up the assertion string by removing some extraneous characters.
   Splitting the assertion into pieces */

String *Contracts::make_expression(String *s, Node *n) {
  String *str_assert, *expr = 0;
  List *list_assert;

  str_assert = NewString(s);
  /* Omit all useless characters and split by ; */
  Replaceall(str_assert, "\n", "");
  Replaceall(str_assert, "{", "");
  Replaceall(str_assert, "}", "");
  Replace(str_assert, " ", "", DOH_REPLACE_ANY | DOH_REPLACE_NOQUOTE);
  Replace(str_assert, "\t", "", DOH_REPLACE_ANY | DOH_REPLACE_NOQUOTE);

  list_assert = Split(str_assert, ';', -1);
  Delete(str_assert);

  /* build up new assertion */
  str_assert = NewString("");
  Iterator ei;

  for (ei = First(list_assert); ei.item; ei = Next(ei)) {
    expr = ei.item;
    if (Len(expr)) {
      Replaceid(expr, Getattr(n, "name"), Swig_cresult_name());
      if (Len(str_assert))
	Append(str_assert, "&&");
      Printf(str_assert, "(%s)", expr);
    }
  }
  Delete(list_assert);
  return str_assert;
}

/* This function substitutes parameter names for argument names in the
   contract specification.  Note: it is assumed that the wrapper code 
   uses arg1 for self and arg2..argn for arguments. */

void Contracts::substitute_parms(String *s, ParmList *p, int method) {
  int argnum = 1;
  char argname[32];

  if (method) {
    Replaceid(s, "$self", "arg1");
    argnum++;
  }
  while (p) {
    sprintf(argname, "arg%d", argnum);
    String *name = Getattr(p, "name");
    if (name) {
      Replaceid(s, name, argname);
    }
    argnum++;
    p = nextSibling(p);
  }
}

int Contracts::emit_contract(Node *n, int method) {
  Hash *contracts;
  Hash *messages;
  String *c;

  ParmList *cparms;

  if (!Getattr(n, "feature:contract"))
    return SWIG_ERROR;

  /* Get contract parameters */
  cparms = Getmeta(Getattr(n, "feature:contract"), "parms");

  /*  Split contract into preassert & postassert */
  contracts = ContractSplit(n);
  if (!contracts)
    return SWIG_ERROR;

  /* This messages hash is used to hold the error messages that will be displayed on
     failed contract. */

  messages = NewHash();

  /* Take the different contract expressions and clean them up a bit */
  Iterator i;
  for (i = First(contracts); i.item; i = Next(i)) {
    String *e = make_expression(i.item, n);
    substitute_parms(e, cparms, method);
    Setattr(contracts, i.key, e);

    /* Make a string containing error messages */
    Setattr(messages, i.key, NewString(e));
  }

  /* If we're in a class. We need to inherit other assertions. */
  if (InClass) {
    inherit_contracts(CurrentClass, n, contracts, messages);
  }

  /* Save information */
  Setattr(n, "contract:rules", contracts);
  Setattr(n, "contract:messages", messages);

  /* Okay.  Generate the contract runtime code. */

  if ((c = Getattr(contracts, "require:"))) {
    Setattr(n, "contract:preassert", NewStringf("SWIG_contract_assert(%s, \"Contract violation: require: %s\");\n", c, Getattr(messages, "require:")));
  }
  if ((c = Getattr(contracts, "ensure:"))) {
    Setattr(n, "contract:postassert", NewStringf("SWIG_contract_assert(%s, \"Contract violation: ensure: %s\");\n", c, Getattr(messages, "ensure:")));
  }
  return SWIG_OK;
}

int Contracts::cDeclaration(Node *n) {
  int ret = SWIG_OK;
  String *decl = Getattr(n, "decl");

  /* Not a function.  Don't even bother with it (for now) */
  if (!SwigType_isfunction(decl))
    return SWIG_OK;

  if (Getattr(n, "feature:contract"))
    ret = emit_contract(n, InClass && !Swig_storage_isstatic(n));
  return ret;
}

int Contracts::constructorDeclaration(Node *n) {
  int ret = SWIG_OK;
  InConstructor = 1;
  if (Getattr(n, "feature:contract"))
    ret = emit_contract(n, 0);
  InConstructor = 0;
  return ret;
}

int Contracts::externDeclaration(Node *n) {
  return emit_children(n);
}

int Contracts::extendDirective(Node *n) {
  return emit_children(n);
}

int Contracts::importDirective(Node *n) {
  return emit_children(n);
}

int Contracts::includeDirective(Node *n) {
  return emit_children(n);
}

int Contracts::namespaceDeclaration(Node *n) {
  return emit_children(n);
}

int Contracts::classDeclaration(Node *n) {
  int ret = SWIG_OK;
  int oldInClass = InClass;
  Node *oldClass = CurrentClass;
  InClass = 1;
  CurrentClass = n;
  emit_children(n);
  InClass = oldInClass;
  CurrentClass = oldClass;
  return ret;
}

int Contracts::top(Node *n) {
  emit_children(n);
  return SWIG_OK;
}
