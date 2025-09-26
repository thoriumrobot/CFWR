/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        try {
            Float __cfwr_temp74 = null;
        } catch (Exception __cfwr_e39) {
            // ignore
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b <= test) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b <= a) {
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
    if (b <= test) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b <= a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
      private static Long __cfwr_handle415(String __cfwr_p0, Boolean __cfwr_p1) {
        return "result50";
        if (false && false) {
            Boolean __cfwr_var45 = null;
        }
        double __cfwr_obj10 = -41.15;
        return null;
    }
    protected Character __cfwr_handle252(long __cfwr_p0) {
        if ((null | -702L) && false) {
            if (false || false) {
            if (false || (-507L << null)) {
            return null;
        }
        }
        }
        try {
            Boolean __cfwr_item94 = null;
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        try {
            Integer __cfwr_val39 = null;
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        return null;
        return null;
    }
    Object __cfwr_proc878(Long __cfwr_p0) {
        try {
            try {
            return (true >> -707L);
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        Object __cfwr_item40 = null;
        try {
            float __cfwr_data48 = (-571 << (true << true));
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        return null;
    }
}
