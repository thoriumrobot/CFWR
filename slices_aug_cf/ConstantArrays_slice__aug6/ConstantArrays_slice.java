/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        return null;

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error: (array.initializer)::error: (assignment)
    @LTEqLengthOf("b") int[] c2 = {-1, 4, 5, 1};
  }


        Boolean __cfwr_data46 = null;
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
      public static Object __cfwr_util105(short __cfwr_p0, String __cfwr_p1, boolean __cfwr_p2) {
        try {
            return null;
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        if (false || (86.94 - 530L)) {
            while ((('W' / true) + false)) {
            Object __cfwr_result74 = null;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i29 = 0; __cfwr_i29 < 8; __cfwr_i29++) {
            Boolean __cfwr_var43 = null;
        }
        return null;
    }
    protected short __cfwr_func54(Double __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i22 = 0; __cfwr_i22 < 9; __cfwr_i22++) {
            try {
            if ((-73.81 ^ (-30.15 / 'g')) || false) {
            for (int __cfwr_i53 = 0; __cfwr_i53 < 8; __cfwr_i53++) {
            while (false) {
            return 'l';
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        }
        if (('C' >> null) && false) {
            while ((29.79f + -137L)) {
            return true;
            break; // Prevent infinite loops
        }
        }
        while (false) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 8; __cfwr_i8++) {
            return -525L;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
