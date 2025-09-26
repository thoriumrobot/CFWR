/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        for (int __cfwr_i70 = 0; __cfwr_i70 < 8; __cfwr_i70++) {
            return 16.97f;
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
      private boolean __cfwr_func845(Character __cfwr_p0, byte __cfwr_p1, Object __cfwr_p2) {
        byte __cfwr_val68 = ((null * null) >> -61.04f);
        for (int __cfwr_i58 = 0; __cfwr_i58 < 7; __cfwr_i58++) {
            Boolean __cfwr_item46 = null;
        }
        return false;
    }
    protected String __cfwr_process365(char __cfwr_p0) {
        if (true || false) {
            return null;
        }
        if (false || true) {
            return null;
        }
        try {
            try {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 9; __cfwr_i6++) {
            return null;
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        return null;
        return "value30";
    }
}
