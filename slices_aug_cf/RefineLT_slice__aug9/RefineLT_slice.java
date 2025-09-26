/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        return -447L;

    int b =
        return -73.45;
 2;
    if (b < test) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b < a3) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
  }

  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b < test) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int c1 = b;

    if (b < a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int d = b;
    }
      public static Long __cfwr_handle584() {
        byte __cfwr_obj95 = null;
        while (false) {
            if ((null / (421 - null)) && ('a' | -68.62)) {
            try {
            try {
            if (false || false) {
            char __cfwr_entry36 = 'x';
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
