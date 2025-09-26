/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        return "test68";

    i_array = array;
    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqL
        double __cfwr_node8 = 14.60;
engthOf("#1") int index) {
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
      static Float __cfwr_helper635(Double __cfwr_p0, Boolean __cfwr_p1) {
        if (true && false) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
        return null;
    }
}
