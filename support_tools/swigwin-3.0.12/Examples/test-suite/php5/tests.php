<?php

// do we have true global vars or just GETSET functions?
// Used to filter out get/set global functions to fake vars...
define(GETSET,1);

$_original_functions=get_defined_functions();
$_original_globals=1;
$_original_classes=get_declared_classes();
$_original_globals=array_keys($GLOBALS);

class check {
  function get_extra_classes($ref=FALSE) {
    static $extra;
    global $_original_classes;
    if ($ref===FALSE) $f=$_original_classes;
    if (! is_array($extra)) {
      $df=array_flip(get_declared_classes());
      foreach($_original_classes as $class) unset($df[$class]);
      $extra=array_keys($df);
    }
    return $extra;
  }

  function get_extra_functions($ref=FALSE,$gs=false) {
    static $extra;
    static $extrags; // for get/setters
    global $_original_functions;
    if ($ref===FALSE) $f=$_original_functions;
    if (! is_array($extra) || $gs) {
      $extra=array();
      $extrags=array();
      $df=get_defined_functions();
      $df=array_flip($df[internal]);
      foreach($_original_functions[internal] as $func) unset($df[$func]);
      // Now chop out any get/set accessors
      foreach(array_keys($df) as $func)
        if ((GETSET && preg_match('/_[gs]et$/', $func)) ||
            preg_match('/^new_/', $func) ||
            preg_match('/_(alter|get)_newobject$/', $func))
          $extrags[]=$func;
        else $extra[]=$func;
//      $extra=array_keys($df);
    }
    if ($gs) return $extrags;
    return $extra;
  }

  function get_extra_globals($ref=FALSE) {
    static $extra;
    global $_original_globals;
    if (! is_array($extra)) {
      if (GETSET) {
        $_extra=array();
        foreach(check::get_extra_functions(false,1) as $global) {
          if (preg_match('/^(.*)_[sg]et$/', $global, $match))
            $_extra[$match[1]] = 1;
        }
        $extra=array_keys($_extra);
      } else {
        if ($ref===FALSE) $ref=$_original_globals;
        if (! is_array($extra)) {
          $df=array_flip(array_keys($GLOBALS));
          foreach($_original_globals as $func) unset($df[$func]);
          // MASK xxxx_LOADED__ variables
          foreach(array_keys($df) as $func)
            if (preg_match('/_LOADED__$/', $func)) unset($df[$func]);
          $extra=array_keys($df);
        }
      }
    }
    return $extra;
  }

  function classname($string,$object) {
    if (!is_object($object))
      return check::fail("The second argument is a " . gettype($object) . ", not an object.");
    if (strtolower($string)!=strtolower($classname=get_class($object))) return check::fail("Object: \$object is of class %s not class %s",$classname,$string);
    return TRUE;
  }

  function classmethods($classname,$methods) {
    if (is_object($classname)) $classname=get_class($classname);
    $classmethods=array_flip(get_class_methods($classname));
    $missing=array();
    $extra=array();
    foreach($methods as $method) {
      if (! isset($classmethods[$method])) $missing[]=$method;
      else unset($classmethods[$method]);
    }
    $extra=array_keys($classmethods);
    if ($missing) $message[]="does not have these methods:\n  ".join(",",$missing);
    if ($message) {
      return check::fail("Class %s %s\nFull class list:\n  %s\n",$classname,join("\nbut ",$message),join("\n  ",get_class_methods($classname)));
    }
    if ($extra) $message[]="Class ".$classname." has these extra methods:\n  ".join(",",$extra);
    if ($message) return check::warn(join("\n  ",$message));
    return TRUE;
  }

  function set($var,$value) {
    $func=$var."_set";
    if (GETSET) $func($value);
    else $_GLOBALS[$var]=$value;
  }

  function &get($var) {
    $func=$var."_get";
    if (GETSET) return $func();
    else return $_GLOBALS[$var];
  }

  function is_a($a,$b) {
    if (is_object($a)) $a=strtolower(get_class($a));
    if (is_object($b)) $a=strtolower(get_class($b));
    $parents=array();
    $c=$a;
    while($c!=$b && $c) {
      $parents[]=$c;
      $c=strtolower(get_parent_class($c));
    }
    if ($c!=$b) return check::fail("Class $a does not inherit from class $b\nHierachy:\n  %s\n",join("\n  ",$parents));
    return TRUE;
  }

  function classparent($a,$b) {
    if (is_object($a)) $a=get_class($a);
    if (is_object($b)) $a=get_class($b);
    $parent=get_parent_class($a);

    if ($parent!=$b) return check::fail("Class $a parent not actually $b but $parent");
    return TRUE;
  }

  function classes($classes) {
    if (! is_array($classes)) $classes=array($classes);
    $message=array();
    $missing=array();
    $extra=array_flip(check::get_extra_classes());
    foreach($classes as $class) {
      if (! class_exists($class)) $missing[]=$class;
      else unset($extra[$class]);
    }
    if ($missing) $message[]=sprintf("Classes missing: %s",join(",",$missing));
    if ($message) return check::fail(join("\n  ",$message));
    if ($extra) $message[]=sprintf("These extra classes are defined: %s",join(",",array_keys($extra)));
    if ($message) return check::warn(join("\n  ",$message));
    return TRUE;    
  }

  function functions($functions) {
    if (! is_array($functions)) $functions=array($functions);
    $message=array();
    $missing=array();
    $extra=array_flip(check::get_extra_functions());

    foreach ($functions as $func) {
      if (! function_exists($func)) $missing[]=$func;
      else unset($extra[$func]);
    }
    if ($missing) $message[]=sprintf("Functions missing: %s",join(",",$missing));
    if ($message) return check::fail(join("\n  ",$message));
    if ($extra) $message[]=sprintf("These extra functions are defined: %s",join(",",array_keys($extra)));
    if ($message) return check::warn(join("\n  ",$message));
    return TRUE;    
  }

  function globals($globals) {
    if (! is_array($globals)) $globals=array($globals);
    $message=array();
    $missing=array();
    $extra=array_flip(check::get_extra_globals());
    foreach ($globals as $glob) {
      if (GETSET) {
        if (! isset($extra[$glob])) $missing[]=$glob;
        else unset($extra[$glob]);
      } else {
        if (! isset($GLOBALS[$glob])) $missing[]=$glob;
        else unset($extra[$glob]);
      }
    }
    if ($missing) $message[]=sprintf("Globals missing: %s",join(",",$missing));
    if ($message) return check::fail(join("\n  ",$message));
    if ($extra) $message[]=sprintf("These extra globals are defined: %s",join(",",array_keys($extra)));
    if ($message) return check::warn(join("\n  ",$message));
    return TRUE;    

  }

  function functionref($a,$type,$message) {
    if (! preg_match("/^_[a-f0-9]+$type$/i", $a))
      return check::fail($message);
    return TRUE;
  }

  function equal($a,$b,$message) {
    if (! ($a===$b)) return check::fail($message . ": '$a'!=='$b'");
    return TRUE;
  }

  function resource($a,$b,$message) {
    $resource=trim(check::var_dump($a));
    if (! preg_match("/^resource\([0-9]+\) of type \($b\)/i", $resource))
      return check::fail($message);
    return TRUE;
  }

  function isnull($a,$message) {
    $value=trim(check::var_dump($a));
    return check::equal($value,"NULL",$message);
  }

  function var_dump($arg) {
    ob_start();
    var_dump($arg);
    $result=ob_get_contents();
    ob_end_clean();
    return $result;
  }

  function fail($pattern) {
    $args=func_get_args();
    print("Failed on: ".call_user_func_array("sprintf",$args)."\n");
    exit(1);
  }

  function warn($pattern) {
    $args=func_get_args();
    print("Warning on: ".call_user_func_array("sprintf",$args)."\n");
    return FALSE;
  }

  function done() {
#    print $_SERVER[argv][0]." ok\n";
  }
}
?>
