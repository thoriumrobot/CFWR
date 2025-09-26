/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        if (true && true) {
            try {
            try {
            while (false) {
            Character __cfwr_entry94 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
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
      protected static double __cfwr_func574(double __cfwr_p0, Float __cfwr_p1, int __cfwr_p2) {
        if (false && (('r' << null) | (false + 27.79f))) {
            return null;
        }
        for (int __cfwr_i29 = 0; __cfwr_i29 < 8; __cfwr_i29++) {
            if (true && true) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 9; __cfwr_i90++) {
            try {
            return 'm';
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
        }
        }
        try {
            short __cfwr_result25 = null;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        if (false || ('n' * true)) {
            while (false) {
            return (-78.24f & -95);
            break; // Prevent infinite loops
        }
        }
        return -0.08;
    }
}
