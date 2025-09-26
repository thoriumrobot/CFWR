/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        if ((false << -93.59f) || false) {
            if (('X' + (53.11 ^ 'v')) || true) {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 4; __cfwr_i22++) {
            while ((false / 29.82)) {
            char __cfwr_entry9 = 'C';
            break; // Prevent infinite loops
        }
        }
        }
        
        try {
            if (false && (false * 359L)) {
            return null;
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
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
      static float __cfwr_util914(Character __cfwr_p0) {
        for (int __cfwr_i79 = 0; __cfwr_i79 < 3; __cfwr_i79++) {
            try {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        }
        if (((true & 56.61f) << -21.18f) || false) {
            while (false) {
            while ((-73.88f * (null - 'Y'))) {
            return ('I' << '2');
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return -25.86f;
    }
}
