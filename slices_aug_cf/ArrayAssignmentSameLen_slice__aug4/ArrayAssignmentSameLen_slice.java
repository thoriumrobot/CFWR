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
      public static long __cfwr_process771(Object __cfwr_p0, Object __cfwr_p1) {
        while (false) {
            try {
            while (false) {
            if (true && false) {
            try {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 2; __cfwr_i75++) {
            while (((null | 4.71) - -12.68f)) {
            float __cfwr_obj62 = 59.16f;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return -200L;
    }
    public int __cfwr_compute321(Integer __cfwr_p0, Object __cfwr_p1) {
        for (int __cfwr_i77 = 0; __cfwr_i77 < 2; __cfwr_i77++) {
            if (true || false) {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 10; __cfwr_i68++) {
            Boolean __cfwr_result60 = null;
        }
        }
        }
        if (false && false) {
            try {
            try {
            if ((94.84 & (-262L ^ -516L)) || (-92.10f ^ ('Q' & null))) {
            return null;
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        }
        try {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 9; __cfwr_i60++) {
            return "result60";
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return -64;
    }
    public static Double __cfwr_handle224(Object __cfwr_p0, Integer __cfwr_p1) {
        long __cfwr_item18 = -208L;
        Long __cfwr_node29 = null;
        try {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 9; __cfwr_i17++) {
            try {
            if (true || ('f' / -472)) {
            while (((-43.00f / 30.85) | -214)) {
            Float __cfwr_entry55 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        try {
            if (false || (528L ^ 99.26)) {
            long __cfwr_item84 = (null / null);
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        return null;
    }
}
