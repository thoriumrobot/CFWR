/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 10; __cfwr_i89++) {
            Integer __cfwr_obj20 = null;
        }

  
        try {
            return -799;
        } catch (Exception __cfwr_e25) {
            // ignore
        }
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
      public static double __cfwr_handle884() {
        return null;
        return 21.52;
    }
    private float __cfwr_proc8() {
        try {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 7; __cfwr_i89++) {
            try {
            try {
            if (false && true) {
            float __cfwr_item91 = ((false >> false) ^ 415);
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        if (true && false) {
            return null;
        }
        return 82.63f;
    }
}
