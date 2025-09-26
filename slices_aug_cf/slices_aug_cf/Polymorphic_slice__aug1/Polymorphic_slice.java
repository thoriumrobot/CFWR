/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        return null;

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
      static int __cfwr_func148() {
        try {
            return -56.72f;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        while (true) {
            try {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 2; __cfwr_i64++) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 3; __cfwr_i49++) {
            if (((true & -654) - 82.28f) && (null << null)) {
            while ((-11 >> null)) {
            try {
            if (false && false) {
            Character __cfwr_var84 = null;
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return 962;
    }
    private boolean __cfwr_helper289(long __cfwr_p0, Object __cfwr_p1, float __cfwr_p2) {
        try {
            if ((null * false) || true) {
            try {
            try {
            Double __cfwr_val42 = null;
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        try {
            return 867;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        try {
            Character __cfwr_val30 = null;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        return 19.24f;
        return true;
    }
}
