/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        for (int __cfwr_i25 = 0; __cfwr_i25 < 1; __cfwr_i25++) {
            return -38.45;
        }

    char[] arr = new char[size];
    arr[index] = val;
  }

  void LessThanOffsetLowerBound(
      int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
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
      private static Float __cfwr_temp378(byte __cfwr_p0, float __cfwr_p1, Object __cfwr_p2) {
        Boolean __cfwr_val50 = null;
        return null;
        for (int __cfwr_i97 = 0; __cfwr_i97 < 5; __cfwr_i97++) {
            char __cfwr_var30 = 'x';
        }
        return null;
    }
}
