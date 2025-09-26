/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        boolean __cfwr_elem65 = true;

    int b = 2;
    if (b < test) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b < a3) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
  }

  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b < test) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int c1 = b;

    if (b < a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int d = b;
    }
      public Float __cfwr_compute489(Integer __cfwr_p0, byte __cfwr_p1, byte __cfwr_p2) {
        Double __cfwr_result54 = null;
        return null;
        return null;
        return null;
    }
}
