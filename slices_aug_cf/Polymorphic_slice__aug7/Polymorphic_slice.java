/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        byte __cfwr_result98 = null;

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
      protected short __cfwr_aux992(double __cfwr_p0, Integer __cfwr_p1, Object __cfwr_p2) {
        if (true && true) {
            if (false && false) {
            while (('U' | null)) {
            Float __cfwr_temp89 = null;
            break; // Prevent infinite loops
        }
        }
        }
        return null;
        return (('e' / null) - (false + null));
    }
    short __cfwr_func410(long __cfwr_p0, char __cfwr_p1, Double __cfwr_p2) {
        while (true) {
            if (((false | -6.22f) | true) || true) {
            Long __cfwr_elem42 = null;
        }
            break; // Prevent infinite loops
        }
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
        try {
            try {
            if (((null * 68.11) / 29.51) && false) {
            return null;
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        return null;
    }
}
