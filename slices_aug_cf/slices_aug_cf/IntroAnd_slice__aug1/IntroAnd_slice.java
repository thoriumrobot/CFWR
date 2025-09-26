/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test_ubc_and(
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        Integer __cfwr_entry52 = null;

    int x = a[i & k];
    int x1 = a[k & i];
    // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.high)
        short __cfwr_obj55 = null;

    int y = a[j & k];
    if (j > -1) {
      int z = a[j & k];
    }
    // :: error: (array.access.unsafe.high)
    int w = a[m & k];
    if (m < a.length) {
      int u = a[m & k];
    }
      static char __cfwr_func839(Long __cfwr_p0) {
        return 49.20f;
        Long __cfwr_temp95 = null;
        return null;
        return (null | null);
    }
}
