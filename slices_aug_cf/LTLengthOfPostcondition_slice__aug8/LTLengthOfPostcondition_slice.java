/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        while (true) {
            return null;
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
      public int __cfwr_temp382(int __cfwr_p0) {
        for (int __cfwr_i70 = 0; __cfwr_i70 < 3; __cfwr_i70++) {
            while (true) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 8; __cfwr_i45++) {
            try {
            int __cfwr_node12 = (null / (-97L / null));
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i16 = 0; __cfwr_i16 < 8; __cfwr_i16++) {
            return '3';
        }
        return 738;
    }
    protected static Character __cfwr_temp179(Double __cfwr_p0) {
        Boolean __cfwr_result83 = null;
        Object __cfwr_val41 = null;
        return null;
    }
    static Integer __cfwr_helper297() {
        try {
            try {
            Character __cfwr_result12 = null;
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        if (true && (null % (606L + 'o'))) {
            while (false) {
            if (true && false) {
            long __cfwr_var45 = -416L;
        }
            break; // Prevent infinite loops
        }
        }
        try {
            if (((537 >> true) / null) || true) {
            Object __cfwr_temp67 = null;
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        return null;
    }
}
