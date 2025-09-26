/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        for (int __cfwr_i22 = 0; __cfwr_i22 < 7; __cfwr_i22++) {
            while (false) {
            for (int __cfwr_i23 = 0; __cfwr_i23 < 10; __cfwr_i23++) {
            return 331;
        }
            break; // Prevent infinite loops
        }
        }

    // :: error: (assignment)
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @LTLengthOf({"arr"}) int b = a;
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @LTEqLengthOf({"arr"}) int i = test;
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
      private static float __cfwr_temp541() {
        return null;
        for (int __cfwr_i43 = 0; __cfwr_i43 < 10; __cfwr_i43++) {
            short __cfwr_val98 = ((-944 << null) ^ -71.05f);
        }
        return null;
        return 99.59f;
    }
    static byte __cfwr_proc870() {
        try {
            try {
            return null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        return null;
        while (false) {
            return -574L;
            break; // Prevent infinite loops
        }
        return null;
        return ((15.30 >> 48.54f) << (-18.77 % -66));
    }
}
