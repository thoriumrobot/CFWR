/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        if (true || (null + null)) {
            return null;
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
      protected static Long __cfwr_handle258(double __cfwr_p0) {
        String __cfwr_result24 = "hello20";
        return null;
    }
    private static Integer __cfwr_func261() {
        return "test67";
        return null;
    }
    private static Object __cfwr_func336(Object __cfwr_p0) {
        for (int __cfwr_i6 = 0; __cfwr_i6 < 3; __cfwr_i6++) {
            float __cfwr_entry58 = 66.02f;
        }
        if (((966L % 1.89f) % -98.84f) && true) {
            double __cfwr_node98 = -33.21;
        }
        for (int __cfwr_i32 = 0; __cfwr_i32 < 10; __cfwr_i32++) {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 2; __cfwr_i93++) {
            try {
            return null;
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
        }
        Float __cfwr_item73 = null;
        return null;
    }
}
