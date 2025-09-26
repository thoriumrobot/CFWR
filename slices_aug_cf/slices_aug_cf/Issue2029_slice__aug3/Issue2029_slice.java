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
      protected static short __cfwr_util378(char __cfwr_p0) {
        for (int __cfwr_i93 = 0; __cfwr_i93 < 3; __cfwr_i93++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        return -404L;
        if (((true << 'o') / 'Z') && (null >> null)) {
            try {
            byte __cfwr_obj76 = ((70.96f & 958) / (null >> '9'));
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        return null;
    }
    private static char __cfwr_aux841() {
        try {
            Integer __cfwr_var15 = null;
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        return 'a';
    }
}
