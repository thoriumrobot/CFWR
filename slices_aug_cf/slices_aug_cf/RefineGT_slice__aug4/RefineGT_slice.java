/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        for (int __cfwr_i58 = 0; __cfwr_i58 < 8; __cfwr_i58++) {
            if (('y' / (294 >> 456)) || false) {
            while (('i' * null)) {
            while (false) {
            try {
            try {
            boolean __cfwr_result42 = false;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }

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
      protected char __cfwr_compute425() {
        if (((true - 44.30) * true) || true) {
            try {
            return null;
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        }
        return (-51.89 & false);
    }
}
