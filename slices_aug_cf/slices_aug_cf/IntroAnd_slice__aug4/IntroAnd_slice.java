/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test_ubc_and(
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        if (true || (-979L | null)) {
            return null;
        }

    int x = a[i & k];
    int x1 = a[k & i];
    // :: error: (array.access.unsafe.low) ::
        for (int __cfwr_i45 = 0; __cfwr_i45 < 4; __cfwr_i45++) {
            long __cfwr_elem26 = -848L;
        }
 error: (array.access.unsafe.high)
    int y = a[j & k];
    if (j > -1) {
      int z = a[j & k];
    }
    // :: error: (array.access.unsafe.high)
    int w = a[m & k];
    if (m < a.length) {
      int u = a[m & k];
    }
      static float __cfwr_func126() {
        return null;
        return 69.21f;
    }
}
