/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test_ubc_and(
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        return null;

    int x = a[i & k];
    int x1 = a[k & i];
    // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.high)
    int y = a[j & k];
    if (j > -1) {
      int z = a[j & k];
    }
    // :: error: (array.access.unsafe.high)
    int w = a[m & k];
    if (m < a.length) {
      int u = a[m & k];
    }
      static String __cfwr_process887(int __cfwr_p0, short __cfwr_p1, Object __cfwr_p2) {
        while (('v' / 0.01f)) {
            try {
            short __cfwr_result71 = (-65.49 << false);
        } catch (Exception __cfwr_e57) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i44 = 0; __cfwr_i44 < 5; __cfwr_i44++) {
            if (true && true) {
            return null;
        }
        }
        try {
            short __cfwr_result24 = null;
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        return "value79";
    }
    protected Float __cfwr_calc324(double __cfwr_p0, double __cfwr_p1) {
        return "result77";
        return null;
    }
    static Double __cfwr_compute457(Boolean __cfwr_p0) {
        try {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 3; __cfwr_i63++) {
            Object __cfwr_var99 = null;
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        Boolean __cfwr_elem51 = null;
        while ((false ^ -940L)) {
            try {
            return -17.57;
        } catch (Exception __cfwr_e48) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i90 = 0; __cfwr_i90 < 2; __cfwr_i90++) {
            int __cfwr_data76 = 127;
        }
        return null;
    }
}
