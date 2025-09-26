/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        for (int __cfwr_i9 = 0; __cfwr_i9 < 7; __cfwr_i9++) {
            while (true) {
            boolean __cfwr_elem9 = false;
            break; // Prevent infinite loops
        }
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // ::
        if (false && false) {
            Float __cfwr_node31 = null;
        }
 error: (assignment)
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
      protected static Character __cfwr_helper55() {
        if (true || false) {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 2; __cfwr_i17++) {
            if (true || false) {
            Long __cfwr_node4 = null;
        }
        }
        }
        try {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 10; __cfwr_i54++) {
            boolean __cfwr_node86 = false;
        }
        } catch (Exception __cfwr_e7) {
            // ignore
        }
        return null;
    }
    private boolean __cfwr_proc154() {
        for (int __cfwr_i31 = 0; __cfwr_i31 < 1; __cfwr_i31++) {
            try {
            return null;
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        }
        if (false || true) {
            for (int __cfwr_i80 = 0; __cfwr_i80 < 6; __cfwr_i80++) {
            try {
            return null;
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        }
        }
        return null;
        Long __cfwr_node42 = null;
        return false;
    }
}
