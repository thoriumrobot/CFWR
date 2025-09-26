/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test_ubc_and(
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        String __cfwr_item33 = "world46";

    int x = a[i & k];
    int x1 = a[k & i];
    // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.hi
        Float __cfwr_obj20 = null;
gh)
    int y = a[j & k];
    if (j > -1) {
      int z = a[j & k];
    }
    // :: error: (array.access.unsafe.high)
    int w = a[m & k];
    if (m < a.length) {
      int u = a[m & k];
    }
      protected double __cfwr_temp621(String __cfwr_p0) {
        for (int __cfwr_i97 = 0; __cfwr_i97 < 2; __cfwr_i97++) {
            try {
            if ((541L ^ false) && true) {
            while (false) {
            if (true && false) {
            return 643L;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        while (false) {
            while (false) {
            long __cfwr_temp25 = -348L;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        while (false) {
            return -197L;
            break; // Prevent infinite loops
        }
        return -58.20;
    }
}
