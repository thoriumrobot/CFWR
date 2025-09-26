/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        try {
            if (true && true) {
            if (true || ((null - null) >> null)) {
            while (false) {
            while (('R' | (null >> 'P'))) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e40) {
            // ignore
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
      Object __cfwr_func145(float __cfwr_p0, short __cfwr_p1) {
        while (false) {
            while (((-17 | 'c') * null)) {
            return 289L;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i60 = 0; __cfwr_i60 < 1; __cfwr_i60++) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 10; __cfwr_i38++) {
            short __cfwr_item34 = ((54L * -12.13) * (-795 << -810L));
        }
        }
        return null;
    }
}
