/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        try {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 3; __cfwr_i26++) {
            int __cfwr_data77 = 872;
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }

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
      public static float __cfwr_func883(int __cfwr_p0, String __cfwr_p1, char __cfwr_p2) {
        if (false && ((-32.26 * 'h') % (-94.04 | 5.30f))) {
            return 564;
        }
        while (true) {
            if (true || true) {
            try {
            Long __cfwr_var34 = null;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i93 = 0; __cfwr_i93 < 9; __cfwr_i93++) {
            int __cfwr_node49 = -700;
        }
        while (true) {
            Float __cfwr_val8 = null;
            break; // Prevent infinite loops
        }
        return ((-55.31f * null) & 655);
    }
    protected static Object __cfwr_process4(short __cfwr_p0, Float __cfwr_p1) {
        return null;
        return null;
    }
}
