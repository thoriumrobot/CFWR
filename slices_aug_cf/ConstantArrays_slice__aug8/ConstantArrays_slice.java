/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        for (int __cfwr_i83 = 0; __cfwr_i83 < 7; __cfwr_i83++) {
            while (true) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 3; __cfwr_i49++) {
            byte __cfwr_obj73 = ('b' << (-27L * -62.00f));
        }
            break; // Prevent infinite loops
        }
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
      public static long __cfwr_func40(Boolean __cfwr_p0, Double __cfwr_p1) {
        return ((-984L / -30.76) + -346);
        try {
            while ((171 * false)) {
            return false;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        return 584L;
    }
    private Character __cfwr_util232(String __cfwr_p0, double __cfwr_p1, double __cfwr_p2) {
        return null;
        for (int __cfwr_i11 = 0; __cfwr_i11 < 5; __cfwr_i11++) {
            if ((621L - 100) && true) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 4; __cfwr_i33++) {
            try {
            Long __cfwr_val44 = null;
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        }
        }
        }
        return null;
    }
    private int __cfwr_handle126(int __cfwr_p0, Long __cfwr_p1) {
        try {
            return ('U' ^ null);
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        char __cfwr_elem1 = 'L';
        return -38;
    }
}
