/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        for (int __cfwr_i25 = 0; __cfwr_i25 < 5; __cfwr_i25++) {
            while (true) {
            try {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 5; __cfwr_i66++) {
            return 705L;
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
            break; // Prevent infinite loops
        }
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
      long __cfwr_calc638() {
        if (true || true) {
            return null;
        }
        if (true && false) {
            return null;
        }
        try {
            if (true || false) {
            while (true) {
            if (false || (null * (-295L % null))) {
            byte __cfwr_val72 = null;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        boolean __cfwr_obj68 = false;
        return 536L;
    }
}
