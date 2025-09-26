/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        Integer __cfwr_node13 = null;

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
      static Character __cfwr_util834() {
        if (true || (null & 'E')) {
            try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 8; __cfwr_i77++) {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 8; __cfwr_i52++) {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 5; __cfwr_i54++) {
            while (false) {
            while (false) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 2; __cfwr_i74++) {
            if (true || false) {
            byte __cfwr_item39 = null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        }
        return null;
    }
    protected byte __cfwr_temp894(Object __cfwr_p0, boolean __cfwr_p1) {
        for (int __cfwr_i52 = 0; __cfwr_i52 < 2; __cfwr_i52++) {
            int __cfwr_result74 = -355;
        }
        return null;
    }
}
