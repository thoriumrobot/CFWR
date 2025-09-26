/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        try {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 4; __cfwr_i5++) {
            return null;
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b <= test) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b <= a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
  }

  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b <= test) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b <= a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
      public static boolean __cfwr_helper645(String __cfwr_p0, byte __cfwr_p1, char __cfwr_p2) {
        try {
            return null;
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        while (false) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 2; __cfwr_i97++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        if (true && true) {
            Long __cfwr_temp69 = null;
        }
        return true;
    }
}
