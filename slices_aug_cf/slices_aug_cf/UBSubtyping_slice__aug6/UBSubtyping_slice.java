/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        while ((('x' ^ 84.46) | null)) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 2; __cfwr_i34++) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 3; __cfwr_i64++) {
            if (false && (true ^ 16.23)) {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 8; __cfwr_i79++) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 6; __cfwr_i55++) {
            Object __cfwr_var88 = null;
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
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
      public float __cfwr_helper833() {
        return null;
        Integer __cfwr_data1 = null;
        Integer __cfwr_val6 = null;
        return 49.20f;
    }
}
