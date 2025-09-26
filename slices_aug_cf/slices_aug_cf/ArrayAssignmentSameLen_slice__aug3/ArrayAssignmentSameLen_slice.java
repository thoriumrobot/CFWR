/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        Integer __cfwr_entry52 = null;

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
      public static short __cfwr_func894(Long __cfwr_p0) {
        return 91.24f;
        while (((null & 50.45f) | null)) {
            return null;
            break; // Prevent infinite loops
        }
        double __cfwr_entry10 = (16.35f / 15L);
        return null;
    }
    private Double __cfwr_process202(Double __cfwr_p0, Double __cfwr_p1, Object __cfwr_p2) {
        for (int __cfwr_i15 = 0; __cfwr_i15 < 10; __cfwr_i15++) {
            if (true && false) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 3; __cfwr_i69++) {
            try {
            while ((12.33f >> (null << 11.50))) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 8; __cfwr_i11++) {
            double __cfwr_node60 = -48.41;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        }
        }
        }
        return "test8";
        try {
            while (false) {
            Double __cfwr_elem73 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        return -893;
        return null;
    }
}
