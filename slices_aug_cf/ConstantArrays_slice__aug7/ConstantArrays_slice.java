/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        try {
            return "value46";
        } catch (Exception __cfwr_e26) {
            // ignore
        }

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error:
        double __cfwr_obj8 = -87.86;
 (array.initializer)::error: (assignment)
    @LTEqLengthOf("b") int[] c2 = {-1, 4, 5, 1};
  }

  void offset_test() {
    int[] b = new int[4];
    int[] b2 = new int[10];
    @LTLengthOf(
        value = {"b", "b2"},
        offset = {"-2", "5"})
    int[] a = {2, 3, 0};

    @LTLengthOf(
        value = {"b", "b2"},
        offset = {"-2", "5"})
    // :: error: (array.initializer)::error: (assignment)
    int[] a2 = {2, 3, 5};

    // Non-constant offsets don't work correctly. See kelloggm#120.
      private static Character __cfwr_temp614(int __cfwr_p0) {
        try {
            try {
            return null;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        return null;
    }
}
