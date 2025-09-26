/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        for (int __cfwr_i50 = 0; __cfwr_i50 < 1; __cfwr_i50++) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 3; __cfwr_i38++) {
            while (('A' >> true)) {
            long __cfwr_elem74 = -372L;
            break; // Prevent infinite loops
        }
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
      static Character __cfwr_helper197() {
        while (true) {
            try {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 3; __cfwr_i10++) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 7; __cfwr_i90++) {
            Integer __cfwr_data51 = null;
        }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        while (false) {
            Boolean __cfwr_var59 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
    private static Double __cfwr_proc230(Integer __cfwr_p0, Boolean __cfwr_p1) {
        Character __cfwr_obj6 = null;
        if ((null | '5') || false) {
            int __cfwr_result70 = (-102 << true);
        }
        return null;
    }
}
