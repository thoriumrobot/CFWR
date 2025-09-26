/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        try {
            if ((null & false) && false) {
            char __cfwr_obj96 = '2';
        }
        } catch (Exception __cfwr_e61) {
            // ignore
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
      private int __cfwr_temp383(Boolean __cfwr_p0, int __cfwr_p1, byte __cfwr_p2) {
        try {
            while (((-25.81f | null) + 28.21f)) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        try {
            try {
            try {
            int __cfwr_obj76 = (-52.55 - (-32.21 & 571));
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        try {
            while (false) {
            if (true || (('Q' ^ null) >> 290)) {
            try {
            if ((true % true) && true) {
            return null;
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        if (('k' % null) || (null ^ -976)) {
            if (false && true) {
            for (int __cfwr_i70 = 0; __cfwr_i70 < 7; __cfwr_i70++) {
            Boolean __cfwr_val38 = null;
        }
        }
        }
        return 551;
    }
    int __cfwr_temp202(double __cfwr_p0, Long __cfwr_p1) {
        double __cfwr_temp26 = -86.43;
        if (true || true) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 1; __cfwr_i55++) {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 8; __cfwr_i68++) {
            try {
            Long __cfwr_item23 = null;
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        }
        }
        }
        if ((true & null) || false) {
            if ((null & 172) && false) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        return -205;
    }
}
