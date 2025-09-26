/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        while (false) {
            try {
            while (false) {
            try {
            Object __cfwr_obj26 = null;
        } catch (Exception __cfwr_e91) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
            break; // Prevent i
        try {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 10; __cfwr_i74++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 10; __cfwr_i61++) {
            try {
            return null;
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
nfinite loops
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
      protected static double __cfwr_proc845(double __cfwr_p0) {
        if (false && (31L + 461L)) {
            Integer __cfwr_elem20 = null;
        }
        char __cfwr_val21 = '3';
        return 33.94;
    }
    public Double __cfwr_util438(double __cfwr_p0) {
        for (int __cfwr_i40 = 0; __cfwr_i40 < 9; __cfwr_i40++) {
            return 75.88;
        }
        return null;
    }
}
