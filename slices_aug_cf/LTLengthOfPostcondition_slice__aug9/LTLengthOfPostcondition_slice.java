/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        while (true) {
            for (int __cfwr_i80 = 0; __cfwr_i80 < 7; __cfwr_i80++) {
            try {
            while (false) {
            byte __cfwr_obj62 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        }
            break; // Prevent infinite loops
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
      Float __cfwr_aux99(Object __cfwr_p0, String __cfwr_p1) {
        int __cfwr_entry40 = 871;
        short __cfwr_temp25 = null;
        if ((-48.06f & null) && false) {
            try {
            while (false) {
            try {
            if (('t' / false) || false) {
            try {
            try {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 6; __cfwr_i52++) {
            try {
            try {
            return ((true ^ 635L) & false);
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        }
        for (int __cfwr_i61 = 0; __cfwr_i61 < 10; __cfwr_i61++) {
            try {
            Object __cfwr_data32 = null;
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        }
        return null;
    }
    static char __cfwr_aux283(float __cfwr_p0) {
        while ((339L / null)) {
            while (false) {
            while (true) {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 1; __cfwr_i99++) {
            while (true) {
            while (((70.79f >> 778) >> (true - 'p'))) {
            if (false || true) {
            for (int __cfwr_i70 = 0; __cfwr_i70 < 4; __cfwr_i70++) {
            while (true) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 6; __cfwr_i77++) {
            return 592;
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i70 = 0; __cfwr_i70 < 7; __cfwr_i70++) {
            double __cfwr_obj76 = -14.34;
        }
        return null;
        return null;
        return 'o';
    }
}
