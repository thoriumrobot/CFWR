/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        double __cfwr_node10 = -68.18;

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (test > b) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (a > b) {
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
    if (test > b) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (a > b) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
      public int __cfwr_aux148(Double __cfwr_p0, Boolean __cfwr_p1) {
        try {
            return 31.29f;
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        if (true || true) {
            Object __cfwr_data66 = null;
        }
        for (int __cfwr_i96 = 0; __cfwr_i96 < 8; __cfwr_i96++) {
            while (false) {
            if (false || (723 * (-883L - true))) {
            try {
            if (false && false) {
            short __cfwr_result52 = null;
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        while (true) {
            if ((-12.90f ^ null) && false) {
            return ((-81.44f << -38.21) ^ (-46.41f * -280L));
        }
            break; // Prevent infinite loops
        }
        return 173;
    }
}
