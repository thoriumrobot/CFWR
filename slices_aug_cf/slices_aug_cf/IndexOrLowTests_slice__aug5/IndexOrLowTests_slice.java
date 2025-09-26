/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        try {
            long __cfwr_elem38 = 161L;
        } catch (Exception __cfwr_e17) {
            // ignore
        }


    if (index != -1) {
      array[index] = 1;
    }

    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    array[y] = 1;
    if (y < array.length) {
      array[y] = 1;
    }
    // :: error: (assignment)
    index = array.length;
      public Float __cfwr_util668(Object __cfwr_p0, int __cfwr_p1, boolean __cfwr_p2) {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 6; __cfwr_i89++) {
            try {
            try {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 7; __cfwr_i15++) {
            boolean __cfwr_data14 = true;
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        }
        try {
            char __cfwr_var70 = 'W';
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        for (int __cfwr_i94 = 0; __cfwr_i94 < 10; __cfwr_i94++) {
            try {
            while (((null ^ -449) << null)) {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 9; __cfwr_i4++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        }
        return null;
    }
}
