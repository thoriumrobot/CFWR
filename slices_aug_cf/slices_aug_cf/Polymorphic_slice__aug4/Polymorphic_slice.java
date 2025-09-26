/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        for (int __cfwr_i43 = 0; __cfwr_i43 < 1; __cfwr_i43++) {
            while (false) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        } catch (Exception __cfwr_e87) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }

    int[] banana;
    int @SameLen("a2") [] b = samelen_identity(a);
    // :: error: (assignment)
    int @SameLen("banana") [] c = samelen_identity(b);
  }

  // UpperBound tests
  void ubc_id(
      int[] a,
      int[] b,
      @LTLengthOf("#1") int ai,
      @LTEqLengthOf("#1") int al,
      @LTLengthOf({"#1", "#2"}) int abi,
      @LTEqLengthOf({"#1", "#2"}) int abl) {
    int[] c;

    @LTLengthOf("a") int ai1 = ubc_identity(ai);
    // :: error: (assignment)
    @LTLengthOf("b") int ai2 = ubc_identity(ai);

    @LTEqLengthOf("a") int al1 = ubc_identity(al);
    // :: error: (assignment)
    @LTLengthOf("a") int al2 = ubc_identity(al);

    @LTLengthOf({"a", "b"}) int abi1 = ubc_identity(abi);
    // :: error: (assignment)
    @LTLengthOf({"a", "b", "c"}) int abi2 = ubc_identity(abi);

    @LTEqLengthOf({"a", "b"}) int abl1 = ubc_identity(abl);
    // :: error: (assignment)
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = ubc_identity(abl);
      protected Long __cfwr_calc350(boolean __cfwr_p0, Boolean __cfwr_p1, boolean __cfwr_p2) {
        while (true) {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 6; __cfwr_i65++) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 1; __cfwr_i88++) {
            if (false || false) {
            return null;
        }
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
    long __cfwr_handle786(Boolean __cfwr_p0, Boolean __cfwr_p1, Boolean __cfwr_p2) {
        while ((null / 17.85)) {
            return null;
            break; // Prevent infinite loops
        }
        long __cfwr_elem98 = (-141L & (null + -61.48));
        if (((821 | null) * 89.53f) || true) {
            while (true) {
            if (false || false) {
            boolean __cfwr_node44 = (-458 + ('X' << true));
        }
            break; // Prevent infinite loops
        }
        }
        return ((true & 94.89) >> false);
    }
}
