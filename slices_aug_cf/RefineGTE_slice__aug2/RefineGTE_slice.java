/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        for (int __cfwr_i63 = 0; __cfwr_i63 < 9; __cfwr_i63++) {
            return null;
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (test >= b) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (a >= b) {
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
    if (test >= b) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int c1 = b;

    if (a >= b) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int d = b;
    }
      public Boolean __cfwr_calc253(Float __cfwr_p0, Double __cfwr_p1, int __cfwr_p2) {
        for (int __cfwr_i20 = 0; __cfwr_i20 < 7; __cfwr_i20++) {
            if (('N' << null) || true) {
            if (true && true) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 7; __cfwr_i20++) {
            while (false) {
            if (true && true) {
            Long __cfwr_val47 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        if (false || true) {
            Boolean __cfwr_result99 = null;
        }
        try {
            if (false && true) {
            while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e49) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        return null;
        return null;
    }
}
