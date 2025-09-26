/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        return "test79";

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
      double __cfwr_process206(short __cfwr_p0, String __cfwr_p1, double __cfwr_p2) {
        if (false && (-188L ^ -148L)) {
            short __cfwr_data12 = null;
        }
        for (int __cfwr_i14 = 0; __cfwr_i14 < 1; __cfwr_i14++) {
            return null;
        }
        if (true || (-1.03 | -4.40)) {
            try {
            float __cfwr_val71 = (('O' + 75.51) | null);
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
        int __cfwr_obj65 = (null & (81.54f - 83.50));
        return 61.06;
    }
    public static Character __cfwr_handle866() {
        Boolean __cfwr_data36 = null;
        return ((757 >> '5') / -2);
        int __cfwr_data46 = -344;
        try {
            Integer __cfwr_entry61 = null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        return null;
    }
}
