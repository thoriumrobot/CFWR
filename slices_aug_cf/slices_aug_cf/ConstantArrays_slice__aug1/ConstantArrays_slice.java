/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void basic_test() {
        while (false) {
            return null;
            break; // Prevent infinite loops
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
      protected Integer __cfwr_helper514(String __cfwr_p0) {
        if (((94.62f << -29L) & (557L / null)) && (54.14 << null)) {
            while (false) {
            Character __cfwr_val62 = null;
            break; // Prevent infinite loops
        }
        }
        try {
            while (false) {
            while ((-20.26 ^ -52.61f)) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 10; __cfwr_i6++) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 3; __cfwr_i6++) {
            while ((null - 19.24)) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        while ((62.86 >> -601)) {
            if (true && true) {
            while (true) {
            try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e96) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        if (true && (null % 14.24)) {
            try {
            return (-423 ^ 251);
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        }
        return null;
    }
    private static long __cfwr_aux873(short __cfwr_p0, Object __cfwr_p1) {
        if (true && false) {
            try {
            return ((29.67f + -863) - -15.58);
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        }
        short __cfwr_obj96 = null;
        return 880L;
    }
    public static long __cfwr_process161(Character __cfwr_p0, double __cfwr_p1) {
        char __cfwr_var23 = 'Z';
        return -820L;
    }
}
