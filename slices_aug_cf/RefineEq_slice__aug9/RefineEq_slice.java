/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testLTL(@LTLengthOf("arr") int test) {
        for (int __cfwr_i2 = 0; __cfwr_i2 < 6; __cfwr_i2++) {
            return "item48";
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
      String __cfwr_process301(float __cfwr_p0, Integer __cfwr_p1) {
        if (true && false) {
            while (true) {
            if (true || false) {
            if (((null & -16.11) | (null + -481L)) || false) {
            try {
            return null;
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        while (true) {
            if (true && (540L & -761)) {
            try {
            if (false && true) {
            try {
            if ((176 ^ 'f') || true) {
            if ((false | -323) || (779L - (-54.01 ^ 799L))) {
            if (false && true) {
            if (false && true) {
            String __cfwr_obj10 = "temp78";
        }
        }
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        try {
            while ((80.75f * (false ^ 16.59))) {
            short __cfwr_temp81 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        return "value76";
    }
}
