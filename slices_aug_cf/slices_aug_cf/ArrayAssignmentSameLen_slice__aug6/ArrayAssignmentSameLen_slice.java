/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        try {
            while (true) {
            if (false || true) {
            if (false || true) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e95) {
            // ignore
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
      double __cfwr_proc568(short __cfwr_p0, Long __cfwr_p1, Float __cfwr_p2) {
        if (true && ((-67 * null) * 'H')) {
            if (((null * 57.86f) ^ (-32.34 << null)) || true) {
            if (false || (58.65f | false)) {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 1; __cfwr_i94++) {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 10; __cfwr_i82++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        }
        for (int __cfwr_i51 = 0; __cfwr_i51 < 4; __cfwr_i51++) {
            try {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 9; __cfwr_i87++) {
            if (true || false) {
            return 878;
        }
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        }
        return 21.30;
    }
    public Float __cfwr_aux590(Object __cfwr_p0) {
        try {
            return 93.38;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        return null;
        return null;
    }
    protected int __cfwr_process629() {
        return ((null * null) >> 16.82f);
        for (int __cfwr_i91 = 0; __cfwr_i91 < 5; __cfwr_i91++) {
            if (true && true) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 6; __cfwr_i55++) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 3; __cfwr_i91++) {
            if (true || (null >> -191)) {
            return null;
        }
        }
        }
        }
        }
        char __cfwr_obj6 = 'c';
        return 372;
    }
}
