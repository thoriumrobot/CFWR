/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        if (true && false) {
            try {
            if (true && false) {
            int __cfwr_val2 = -563;
        }
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        }

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @LTLengthOf("b")
        return null;
 int[] a1 = {0, 1, 2, 4};

    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error: (array.initializer)::error: (assignment)
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
      private double __cfwr_helper253() {
        if (((-336 >> -34.21f) + 'd') && false) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 8; __cfwr_i45++) {
            return 448L;
        }
        }
        for (int __cfwr_i41 = 0; __cfwr_i41 < 9; __cfwr_i41++) {
            return null;
        }
        Float __cfwr_elem44 = null;
        return (null & -431L);
    }
}
