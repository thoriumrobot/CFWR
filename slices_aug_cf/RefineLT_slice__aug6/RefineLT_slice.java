/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        try {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 5; __cfwr_i24++) {
            if (false || true) {
            if (true && true) {
            float __cfwr_entry65 = 54.01f;
        }
        }
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }

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
      boolean __cfwr_temp882(String __cfwr_p0, String __cfwr_p1) {
        long __cfwr_node10 = -544L;
        return null;
        return false;
    }
    protected static int __cfwr_temp860(char __cfwr_p0) {
        Boolean __cfwr_item85 = null;
        return -880;
    }
    protected static Double __cfwr_proc595(Boolean __cfwr_p0) {
        return 375;
        return null;
        for (int __cfwr_i85 = 0; __cfwr_i85 < 8; __cfwr_i85++) {
            if ((-326L >> (-223L >> -500L)) && false) {
            Float __cfwr_node11 = null;
        }
        }
        for (int __cfwr_i9 = 0; __cfwr_i9 < 3; __cfwr_i9++) {
            if (((-91.48 << null) << 255L) || (('W' | -308) / true)) {
            return "temp98";
        }
        }
        return null;
    }
}
