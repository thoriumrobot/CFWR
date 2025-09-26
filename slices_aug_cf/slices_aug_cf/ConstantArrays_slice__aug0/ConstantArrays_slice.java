/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        if ((false + false) || false) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 7; __cfwr_i84++) {
            while (true) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 5; __cfwr_i47++) {
            try {
            while ((-3.48 << false)) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
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
      private static String __cfwr_calc740(Integer __cfwr_p0, Float __cfwr_p1) {
        try {
            try {
            while (false) {
            for (int __cfwr_i2 = 0; __cfwr_i2 < 4; __cfwr_i2++) {
            return 968;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        if (false && false) {
            try {
            try {
            try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 3; __cfwr_i32++) {
            return 134L;
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        }
        try {
            Object __cfwr_val41 = null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        while (((-756L * -854) % -170L)) {
            return null;
            break; // Prevent infinite loops
        }
        return "value94";
    }
}
