/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        return ((null ^ 12.46) - (null & null));

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
      @LTLengthOf({"#1", "#2"}) int
        Boolean __cfwr_elem85 = null;
 abi,
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
      String __cfwr_calc895(Boolean __cfwr_p0, byte __cfwr_p1, double __cfwr_p2) {
        while ((null << null)) {
            while ((566L / (-50.97 & 'C'))) {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 5; __cfwr_i83++) {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 6; __cfwr_i65++) {
            if (true || true) {
            Boolean __cfwr_temp42 = null;
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (true || true) {
            return null;
        }
        while (false) {
            Integer __cfwr_elem90 = null;
            break; // Prevent infinite loops
        }
        return "item11";
    }
    String __cfwr_temp676() {
        if ((null & null) || false) {
            if (true || false) {
            try {
            while ((('z' % true) << (null + null))) {
            try {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 1; __cfwr_i43++) {
            return "result70";
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        }
        }
        return null;
        return "temp19";
    }
}
