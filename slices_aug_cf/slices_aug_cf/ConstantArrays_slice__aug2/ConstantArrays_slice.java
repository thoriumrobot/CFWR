/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        while (false) {
            try {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 10; __cfwr_i59++) {
            try {
            Double __cfwr_item48 = null;
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
            break; // Prevent inf
        for (int __cfwr_i77 = 0; __cfwr_i77 < 4; __cfwr_i77++) {
            return 42;
        }
inite loops
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
      Long __cfwr_calc127(Boolean __cfwr_p0) {
        try {
            try {
            Character __cfwr_var6 = null;
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        for (int __cfwr_i34 = 0; __cfwr_i34 < 3; __cfwr_i34++) {
            float __cfwr_data61 = -59.02f;
        }
        try {
            try {
            if ((178L | -585L) && true) {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 4; __cfwr_i22++) {
            Integer __cfwr_obj5 = null;
        }
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        return null;
    }
    public static short __cfwr_util828(long __cfwr_p0) {
        for (int __cfwr_i68 = 0; __cfwr_i68 < 8; __cfwr_i68++) {
            if (false || (null >> (-394L & null))) {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 3; __cfwr_i51++) {
            try {
            if (true || true) {
            try {
            if (true && false) {
            return null;
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        }
        }
        return null;
    }
}
