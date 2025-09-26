/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        Double __cfwr_val15 = null;

    i_array = array;
    i_index = index;
  }

  void test1(int[] a, int[
        try {
            return null;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
] b, @LTEqLengthOf("#1") int index) {
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
      public static char __cfwr_temp770() {
        int __cfwr_obj58 = -17;
        for (int __cfwr_i84 = 0; __cfwr_i84 < 2; __cfwr_i84++) {
            return null;
        }
        return 'I';
    }
    public static Character __cfwr_aux901(String __cfwr_p0, Long __cfwr_p1, double __cfwr_p2) {
        for (int __cfwr_i26 = 0; __cfwr_i26 < 10; __cfwr_i26++) {
            try {
            while (false) {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 9; __cfwr_i13++) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 9; __cfwr_i33++) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 6; __cfwr_i87++) {
            if (((null ^ 3.49) | ('C' << -279)) || false) {
            try {
            return null;
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        }
        return null;
    }
}
