/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        try {
            return (null | 'O');
        } catch (Exception __cfwr_e40) {
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
      protected static short __cfwr_compute465(byte __cfwr_p0, Boolean __cfwr_p1, Float __cfwr_p2) {
        for (int __cfwr_i83 = 0; __cfwr_i83 < 7; __cfwr_i83++) {
            try {
            return true;
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        }
        return (false * 58.24);
    }
    protected static float __cfwr_func630(String __cfwr_p0, String __cfwr_p1) {
        if (true || false) {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 6; __cfwr_i66++) {
            double __cfwr_result98 = (-108 % (null * null));
        }
        }
        try {
            return -247L;
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        long __cfwr_var22 = -319L;
        return -42.51f;
    }
    Integer __cfwr_helper281(boolean __cfwr_p0) {
        if (true && false) {
            if ((-23.89 | -55.74f) || false) {
            while ((18.18f / (null - null))) {
            try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 7; __cfwr_i57++) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 8; __cfwr_i74++) {
            try {
            int __cfwr_item46 = ((true & 16.77) + null);
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        if ((true << 'i') || (null + null)) {
            if ((null + (169 / false)) || false) {
            return null;
        }
        }
        Float __cfwr_data29 = null;
        return null;
    }
}
