/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        return 90.22;

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
      protected char __cfwr_temp528() {
        return null;
        Boolean __cfwr_result24 = null;
        try {
            Float __cfwr_entry12 = null;
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        return 'K';
    }
    private Object __cfwr_func338(Boolean __cfwr_p0) {
        return null;
        int __cfwr_result15 = -646;
        return ((-743L - null) | null);
        return null;
    }
}
