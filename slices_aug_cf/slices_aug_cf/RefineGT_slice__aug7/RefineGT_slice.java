/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        boolean __cfwr_temp26 = true;

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3
        return null;
");

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
      private static float __cfwr_func500(int __cfwr_p0) {
        for (int __cfwr_i19 = 0; __cfwr_i19 < 4; __cfwr_i19++) {
            float __cfwr_obj33 = 45.61f;
        }
        for (int __cfwr_i62 = 0; __cfwr_i62 < 5; __cfwr_i62++) {
            while (false) {
            while (false) {
            if ((null >> (null + 51)) && true) {
            int __cfwr_item87 = (664L + null);
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        if (true && false) {
            try {
            if (('9' | -82.11f) && true) {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 3; __cfwr_i52++) {
            return "hello24";
        }
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        }
        return 63.92f;
    }
}
