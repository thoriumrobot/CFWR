/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        return null;

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test == b) {
      @LTLengthOf("arr") int c = b;

    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int e = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int d = b;
  }

  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test == b) {
      @LTEqLengthOf("arr") int c = b;

      @LTLengthOf("arr") int g = b;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int e = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int d = b;
      private int __cfwr_aux45(String __cfwr_p0, char __cfwr_p1) {
        return null;
        try {
            try {
            if (false && false) {
            return null;
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        return 4;
    }
}
