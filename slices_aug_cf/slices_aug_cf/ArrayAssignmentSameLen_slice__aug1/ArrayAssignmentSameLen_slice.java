/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        return null;

    i_array = array;
    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
    int[] array = a;
    @LTLengthOf(
        value = {"array", "b"},
        offset = {"0", "-3"})
    // :: error: (assignment)
    int i = index;
  }

  void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
    int[] c = a;
    // :: error: (assignment)
    @LTLengthOf(value = {"c", "b"}) int x = i;
    @LTLengthOf("c") int y = i;
  }

  void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {
    int[] c1 = a;
    // See useTest3 for an example of why this assignment should fail.
    @LTLengthOf(
        value = {"c1", "c1"},
        offset = {"0", "x"})
    // :: error: (assignment)
    int z = i;
      protected static float __cfwr_process13(long __cfwr_p0) {
        return null;
        return 88.73f;
    }
    private static float __cfwr_temp847() {
        if (((null * 'M') >> 'X') && (731L - true)) {
            if (((-769 % 75.28f) << false) || true) {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 9; __cfwr_i12++) {
            return null;
        }
        }
        }
        return 62.02f;
    }
    static long __cfwr_temp715(Long __cfwr_p0, int __cfwr_p1, Integer __cfwr_p2) {
        if (true || (-18.05 | null)) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        }
        if (false && false) {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 7; __cfwr_i83++) {
            while (true) {
            if ((596L - null) && false) {
            int __cfwr_elem87 = 43;
        }
            break; // Prevent infinite loops
        }
        }
        }
        return 808L;
    }
}
