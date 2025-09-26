/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test_ubc_and(
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        try {
            while (false) {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 8; __cfwr_i71++) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 3; __cfwr_i14++) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }

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
      protected static Integer __cfwr_handle727(Integer __cfwr_p0, double __cfwr_p1) {
        return null;
        float __cfwr_data92 = (-77.17f * 809);
        if (true && (-698L - -64.55)) {
            Float __cfwr_item74 = null;
        }
        return null;
    }
}
