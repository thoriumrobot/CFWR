/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        for (int __cfwr_i54 = 0; __cfwr_i54 < 8; __cfwr_i54++) {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 2; __cfwr_i98++) {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 6; __cfwr_i63++) {
            Float __cfwr_temp29 = null;
        }
        }
        }

    // :: error: (argument)
    Arrays.fill(array, end, end + x, null);
    shiftIndex(x);
    Arrays.fill(array, end, end + x, null);
  }

  @EnsuresLTLengthOfIf(expression = "end", result = true, targetValue = "array", offset = "#1 - 1")
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
      public static String __cfwr_aux866(short __cfwr_p0, float __cfwr_p1) {
        if (false && false) {
            for (int __cfwr_i18 = 0; __cfwr_i18 < 9; __cfwr_i18++) {
            if (true || false) {
            try {
            return null;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        }
        }
        }
        while ((null << -316L)) {
            try {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 4; __cfwr_i17++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 10; __cfwr_i61++) {
            float __cfwr_var9 = 10.93f;
        }
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        boolean __cfwr_temp37 = ((-17.33f >> -364) >> null);
        return "test22";
    }
    protected char __cfwr_proc289(long __cfwr_p0, Character __cfwr_p1, Boolean __cfwr_p2) {
        while ((3 << -10.50f)) {
            if (false || (null * (true & -686))) {
            return null;
        }
            break; // Prevent infinite loops
        }
        if (false || true) {
            return -6.62f;
        }
        while (false) {
            int __cfwr_val87 = (158 + (540 / null));
            break; // Prevent infinite loops
        }
        return ((true << null) | 35.66);
    }
}
