%module register_par

// bug # 924413
%inline {
  void clear_tree_flags(register struct tree *tp, register int i) {}
}
