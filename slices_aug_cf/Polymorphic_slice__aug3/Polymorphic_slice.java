/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        return 33.95f;

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
      public static Double __cfwr_aux262(long __cfwr_p0) {
        try {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 10; __cfwr_i63++) {
            while ((52.13 - (-5.26f * 57.52))) {
            while (true) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 7; __cfwr_i14++) {
            try {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 1; __cfwr_i43++) {
            while (false) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 10; __cfwr_i78++) {
            return "value52";
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        if (true && false) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 7; __cfwr_i29++) {
            Integer __cfwr_temp99 = null;
        }
        }
        Long __cfwr_item20 = null;
        if (false || false) {
            byte __cfwr_node14 = null;
        }
        return null;
    }
}
