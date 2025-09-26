/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        return null;

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
      public static int __cfwr_process445(Double __cfwr_p0, double __cfwr_p1, long __cfwr_p2) {
        while (true) {
            byte __cfwr_entry43 = null;
            break; // Prevent infinite loops
        }
        return -948;
    }
    private static Double __cfwr_process58(double __cfwr_p0, Object __cfwr_p1) {
        return null;
        return null;
    }
}
