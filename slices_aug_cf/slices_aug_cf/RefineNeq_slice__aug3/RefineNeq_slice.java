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
      Object __cfwr_helper995(Character __cfwr_p0, boolean __cfwr_p1, double __cfwr_p2) {
        while (true) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 1; __cfwr_i89++) {
            short __cfwr_obj90 = (667L & 225);
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    static Object __cfwr_compute824(Long __cfwr_p0) {
        if (false && true) {
            try {
            if (false && false) {
            return (29.62 - 25.31f);
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        }
        try {
            return null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        return ('o' / ('f' >> 36.34f));
        return null;
    }
}
