/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        try {
            if ((45.63 - 'I') && false) {
            try {
            char __cfwr_data80 = 'x';
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }

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
      private static byte __cfwr_handle816(char __cfwr_p0, float __cfwr_p1) {
        return null;
        return null;
    }
    public short __cfwr_compute616(float __cfwr_p0) {
        float __cfwr_val68 = ((94.97f >> null) % ('j' << 303));
        byte __cfwr_elem18 = ((95.40 | 8.06) + (-441 * 776L));
        return null;
    }
}
