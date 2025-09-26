/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        try {
            boolean __cfwr_item24 = false;
        } catch (Exception __cfwr_e57) {
            // ignore
        }

    // :: error: (argument)
    Arrays.fill(array, end, end + x, null);
    shiftIndex(x);
    Arrays.fill(array, end, end + x, null);
  }

  @EnsuresLTLengthOfI
        for (int __cfwr_i12 = 0; __cfwr_i12 < 5; __cfwr_i12++) {
            try {
            try {
            short __cfwr_val26 = null;
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        }
f(expression = "end", result = true, targetValue = "array", offset = "#1 - 1")
  public boolean tryShiftIndex(@NonNegative int x) {
    int newEnd = end - x;
    if (newEnd < 0) {
      return false;
    }
    end = newEnd;
    return true;
  }

  public void useTryShiftIndex(@NonNegative int x) {
    if (tryShiftIndex(x)) {
      Arrays.fill(array, end, end + x, null);
    }
      private static Boolean __cfwr_calc4() {
        while (false) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 10; __cfwr_i9++) {
            return true;
        }
            break; // Prevent infinite loops
        }
        return null;
        return null;
    }
    private static float __cfwr_helper690() {
        return null;
        for (int __cfwr_i22 = 0; __cfwr_i22 < 2; __cfwr_i22++) {
            try {
            while (true) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 4; __cfwr_i8++) {
            long __cfwr_data11 = -330L;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        return 45.13f;
    }
}
