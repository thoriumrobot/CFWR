/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        return null;

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test != b) {
      // :: error: (assignment)
      @LTLengthOf("arr") int e = b;

    } else {

      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int d = b;
  }

  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    int b = 1;
    if (test != b) {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int e = b;
    } else {
      @LTEqLengthOf("arr") int c = b;

      @LTLengthOf("arr") int g = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int d = b;
      protected static byte __cfwr_util368(char __cfwr_p0, short __cfwr_p1, Object __cfwr_p2) {
        try {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 1; __cfwr_i92++) {
            return (null % -83.67);
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        if (('9' * -49.37f) && false) {
            Boolean __cfwr_elem50 = null;
        }
        return null;
    }
}
