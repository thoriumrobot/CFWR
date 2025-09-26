/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        boolean __cfwr_result96 = (3.13 | null);

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
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
      protected int __cfwr_util95(short __cfwr_p0, Double __cfwr_p1) {
        long __cfwr_node53 = -867L;
        try {
            if (true || true) {
            return null;
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        return null;
        return 233;
    }
    boolean __cfwr_proc953(Integer __cfwr_p0) {
        for (int __cfwr_i96 = 0; __cfwr_i96 < 8; __cfwr_i96++) {
            return 'A';
        }
        return (-13.31f - '5');
        return false;
    }
}
