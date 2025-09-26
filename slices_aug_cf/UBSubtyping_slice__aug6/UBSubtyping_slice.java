/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        for (int __cfwr_i6 = 0; __cfwr_i6 < 3; __cfwr_i6++) {
            Object __cfwr_elem76 = null;
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
      protected int __cfwr_proc236() {
        return 704;
        return null;
        return -692;
    }
    protected boolean __cfwr_aux337(long __cfwr_p0, Object __cfwr_p1, byte __cfwr_p2) {
        try {
            while ((629L & '6')) {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 6; __cfwr_i95++) {
            if ((-68L / true) && true) {
            if ((-758 + false) && false) {
            while (true) {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 9; __cfwr_i63++) {
            if ((20.86 | 453) && false) {
            return -73.33f;
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        return true;
    }
}
