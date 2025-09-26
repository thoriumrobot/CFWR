/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        return -740L;

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (test >= b) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (a >= b) {
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
    if (test >= b) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int c1 = b;

    if (a >= b) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int d = b;
    }
      protected static Object __cfwr_process124() {
        for (int __cfwr_i53 = 0; __cfwr_i53 < 9; __cfwr_i53++) {
            while (false) {
            float __cfwr_elem31 = -65.40f;
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    protected long __cfwr_aux372(Double __cfwr_p0, Boolean __cfwr_p1) {
        byte __cfwr_data20 = ((null | null) - false);
        if (((-404 | -769L) / true) || false) {
            try {
            try {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 8; __cfwr_i33++) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 10; __cfwr_i29++) {
            if (true && true) {
            if (true || (('R' ^ 358L) / false)) {
            try {
            if (false || true) {
            try {
            while (false) {
            boolean __cfwr_obj88 = false;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        }
        return -579L;
    }
}
