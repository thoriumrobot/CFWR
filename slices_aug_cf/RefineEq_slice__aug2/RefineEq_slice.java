/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        double __cfwr_val25 = 64.29;

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
      protected long __cfwr_aux956(Integer __cfwr_p0, float __cfwr_p1, Double __cfwr_p2) {
        if (true && (-23.74 | 38.18)) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 2; __cfwr_i59++) {
            return null;
        }
        }
        return null;
        return -777L;
    }
}
