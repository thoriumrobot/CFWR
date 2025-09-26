/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test_ubc_and(
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        for (int __cfwr_i15 = 0; __cfwr_i15 < 5; __cfwr_i15++) {
            try {
            if (true || false) {
            String __cfwr_item59 = "test9";
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
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
      private static Object __cfwr_proc537() {
        while (false) {
            if ((null << true) || (430 - (false - -58.40))) {
            return null;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i11 = 0; __cfwr_i11 < 3; __cfwr_i11++) {
            long __cfwr_elem6 = -438L;
        }
        return ((null & 'X') & 979);
        return null;
    }
}
