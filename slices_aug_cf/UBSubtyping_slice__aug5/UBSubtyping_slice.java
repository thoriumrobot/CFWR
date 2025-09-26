/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        if ((null >> 45.96) && false) {
            if (true && true) {
            Integer __cfwr_val38 = null;
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

    @LTEqLengthOf({"arr", "arr2"}) int h2 = te
        try {
            if ((-41.68 ^ (-57.29f & false)) && true) {
            return "value93";
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
st;
    @LTEqLengthOf({"arr"}) int i = test;
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
      protected int __cfwr_handle617() {
        for (int __cfwr_i39 = 0; __cfwr_i39 < 3; __cfwr_i39++) {
            try {
            return 17.77;
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        }
        if (false || true) {
            while (false) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 2; __cfwr_i78++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        return 39;
    }
}
