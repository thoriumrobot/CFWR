/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        if (true || false) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 9; __cfwr_i8++) {
            try {
            try {
            boolean __cfwr_result62 = (true << 4.56);
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
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
      static Integer __cfwr_helper507(float __cfwr_p0) {
        return false;
        return null;
    }
}
