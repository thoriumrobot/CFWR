/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        while (false) {
            try {
            return ((false >> true) / (-118L << null));
        } catch (Exception __cfwr_e44) {
            // ignore
        }
            break; // Prevent infinite loops
        }

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
      private static Float __cfwr_proc451() {
        Long __cfwr_result35 = null;
        return null;
        if (false && (737L ^ null)) {
            while (false) {
            String __cfwr_node92 = "test50";
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    private static Object __cfwr_temp167(Object __cfwr_p0, double __cfwr_p1) {
        return (-59.96f ^ false);
        for (int __cfwr_i84 = 0; __cfwr_i84 < 6; __cfwr_i84++) {
            return null;
        }
        return null;
    }
    private static char __cfwr_compute120(Double __cfwr_p0, Float __cfwr_p1) {
        while (true) {
            return (null / (true % 4.41));
            break; // Prevent infinite loops
        }
        Boolean __cfwr_val52 = null;
        for (int __cfwr_i56 = 0; __cfwr_i56 < 8; __cfwr_i56++) {
            float __cfwr_data49 = -5.28f;
        }
        return (null / 265);
    }
}
