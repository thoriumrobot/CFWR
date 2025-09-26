/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        for (int __cfwr_i52 = 0; __cfwr_i52 < 2; __cfwr_i52++) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 1; __cfwr_i38++) {
            try {
            if ((-191L & null) || true) {
            while (true) {
            try {
            while ((('O' >> null) * -68.41f)) {
     
        char __cfwr_temp44 = (false / false);
       if (false || false) {
            Object __cfwr_entry6 = null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e42) {
            // ignore
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
      Float __cfwr_calc976() {
        if (false && ((true / 990) + -55.86)) {
            while (true) {
            Character __cfwr_elem79 = null;
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}
