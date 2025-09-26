/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        boolean __cfwr_var4 = ('4' >> -144L);

    char[] arr = new char[size];
    arr[index] = val;
  }

  void LessThanOffsetLowerBound(
      int[] array, @NonNegative @LTLengthOf("#1") i
        try {
            try {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 7; __cfwr_i84++) {
            return null;
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
nt n, @NonNegative @LessThan("#2 + 1") int k) {
    array[n - k] = 10;
  }

  void LessThanOffsetUpperBound(
      @NonNegative int n,
      @NonNegative @LessThan("#1 + 1") int k,
      @NonNegative int size,
      @NonNegative @LessThan("#3 + 1") int index) {
    @NonNegative int m = n - k;
    int[] arr = new int[size];
    // :: error: (unary.increment)
    for (; index < arr.length - 1; index++) {
      arr[index] = 10;
    }
      protected float __cfwr_proc314(Long __cfwr_p0, char __cfwr_p1) {
        return null;
        return -53.75f;
    }
}
