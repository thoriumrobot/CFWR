/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        short __cfwr_item76 = null;

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3")
        Double __cfwr_val69 = null;
;

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
      private long __cfwr_temp329(double __cfwr_p0) {
        return -415;
        while (false) {
            try {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 6; __cfwr_i82++) {
            try {
            if (false || ('1' & 54.62)) {
            Integer __cfwr_entry44 = null;
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return -390L;
    }
}
