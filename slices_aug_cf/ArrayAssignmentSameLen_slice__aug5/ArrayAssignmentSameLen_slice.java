/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        Character __cfwr_val3 = null;

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
      protected boolean __cfwr_calc374(String __cfwr_p0) {
        try {
            while (((-226 * -2.16f) ^ 92.16)) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        return true;
        return ('D' & true);
        short __cfwr_elem44 = null;
        return ((-741L / -828) & (null + false));
    }
    private static Float __cfwr_func820(int __cfwr_p0, Integer __cfwr_p1, char __cfwr_p2) {
        if (true && true) {
            return -521;
        }
        return null;
    }
}
