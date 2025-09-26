/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        Float __cfwr_data19 = null;

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
      private Object __cfwr_process987() {
        try {
            short __cfwr_temp76 = ('h' % null);
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        Object __cfwr_obj62 = null;
        for (int __cfwr_i46 = 0; __cfwr_i46 < 8; __cfwr_i46++) {
            if (true && true) {
            short __cfwr_result25 = null;
        }
        }
        return null;
    }
    private Long __cfwr_compute678(boolean __cfwr_p0, byte __cfwr_p1, short __cfwr_p2) {
        return null;
        if (((974 ^ -85.17) + (null & 'l')) && false) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 4; __cfwr_i47++) {
            try {
            return null;
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        }
        while (true) {
            try {
            return (963L / null);
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
