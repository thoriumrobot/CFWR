/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        for (int __cfwr_i22 = 0; __cfwr_i22 < 9; __cfwr_i22++) {
            while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e88) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: erro
        return null;
r: (array.initializer)::error: (assignment)
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

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
      protected static char __cfwr_handle811(String __cfwr_p0, char __cfwr_p1, char __cfwr_p2) {
        while (false) {
            Integer __cfwr_val58 = null;
            break; // Prevent infinite loops
        }
        return (-540 & null);
    }
}
