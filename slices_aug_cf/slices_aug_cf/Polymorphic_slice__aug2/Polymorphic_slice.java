/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        while (true) {
            if ((-435 >> null) && true) {
            if (false && false) {
            if (true && false) {
            if (false && false) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 10; __cfwr_i49++) {
            if (false && ((14.27 - 90.30f) * false)) {
            try {
            try {
            short __cfwr_temp98 = null;
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
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
      public boolean __cfwr_handle401(int __cfwr_p0, Boolean __cfwr_p1, String __cfwr_p2) {
        Integer __cfwr_obj9 = null;
        return null;
        return false;
    }
}
