/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        return true;

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
      public static int __cfwr_compute299(Integer __cfwr_p0) {
        if (false || false) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 8; __cfwr_i78++) {
            while ((6.44f << '7')) {
            Object __cfwr_data99 = null;
            break; // Prevent infinite loops
        }
        }
        }
        return 598;
    }
    Character __cfwr_process597(double __cfwr_p0) {
        try {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 6; __cfwr_i51++) {
            Boolean __cfwr_data83 = null;
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        try {
            while ((-624 & (77.80 >> null))) {
            while (true) {
            return 207;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        return null;
    }
}
