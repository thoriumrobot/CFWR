/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        String __cfwr_elem84 = "hello7";

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
  
        try {
            try {
            if ((false % (-82.86 - -39.37f)) && true) {
            return null;
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
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
      public static int __cfwr_aux594(Boolean __cfwr_p0, Integer __cfwr_p1) {
        if (false && ((725L % 144) + true)) {
            if (false || (-501 % (true & 'D'))) {
            if (false && false) {
            if (true && false) {
            if (('g' << 'W') && false) {
            try {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 5; __cfwr_i99++) {
            while (('i' << -48.22)) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 8; __cfwr_i8++) {
            while (true) {
            Character __cfwr_item36 = null;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        }
        }
        }
        }
        return 815;
    }
}
