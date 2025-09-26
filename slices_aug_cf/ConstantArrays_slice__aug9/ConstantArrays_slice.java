/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        while (true) {
            Object __cfwr_val90 = null;
            break; // Prevent infinite loops
        }

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error
        for (int __cfwr_i40 = 0; __cfwr_i40 < 5; __cfwr_i40++) {
            while (false) {
            while (true) {
            while (((540L >> 'Q') | (null | false))) {
            while (false) {
            while (true) {
            while (((null - 660) - false)) {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 5; __cfwr_i99++) {
            try {
            return -20.08f;
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
: (array.initializer)::error: (assignment)
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
      public static Character __cfwr_calc55(char __cfwr_p0, Double __cfwr_p1) {
        while (false) {
            long __cfwr_var54 = (('0' % -738L) >> (null / 'O'));
            break; // Prevent infinite loops
        }
        while (((14.37 % true) >> 334)) {
            Long __cfwr_result72 = null;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i82 = 0; __cfwr_i82 < 10; __cfwr_i82++) {
            if (true || true) {
            try {
            try {
            return (null | 40.66f);
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        }
        }
        if (false && true) {
            Character __cfwr_obj55 = null;
        }
        return null;
    }
}
