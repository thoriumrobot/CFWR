/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        return 409L;

    // :: error: (argument)
    Arrays.fill(array, end, end + x, null);
    shiftIndex(x);
    Arrays.fill(array, end, end + x, null);
  }

  @EnsuresLTLengthOfIf(expression = "end", result = true, targetValue = "array", offset = "#1 - 1")
  public boolean tryShiftIndex
        try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 8; __cfwr_i12++) {
            return null;
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
(@NonNegative int x) {
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
      public static Object __cfwr_handle73(Object __cfwr_p0) {
        byte __cfwr_node74 = null;
        return null;
    }
    static long __cfwr_handle628(Integer __cfwr_p0, Double __cfwr_p1, Object __cfwr_p2) {
        return ((-93.80f * 'B') + 88.48);
        String __cfwr_node80 = "hello84";
        return -243L;
    }
    static int __cfwr_util486(short __cfwr_p0, int __cfwr_p1) {
        try {
            while (false) {
            try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 3; __cfwr_i62++) {
            return null;
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        if (false || ('Q' * ('d' >> null))) {
            while ((-284L * (-700L | 66.67))) {
            return 3.13;
            break; // Prevent infinite loops
        }
        }
        if ((62.07f + false) || false) {
            try {
            try {
            String __cfwr_elem56 = "result17";
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
        int __cfwr_elem53 = 592;
        return -854;
    }
}
