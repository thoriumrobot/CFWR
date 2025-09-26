/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        while (true) {
            while (false) {
            if (true && true) {
            try {
            while (true) {
            Character __cfwr_temp31 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }

    // Refines i <= array.length to i < array.length
    if (i != array.length) {
      refineNeqLengthMOne(array, i);
    }
    // No refinement
    if (i != array.length - 1) {
      // :: error: (argument)
      refineNeqLengthMOne(array, i);
    }
  }

  void refineNeqLengthMOne(int[] array, @IndexFor("#1") int i) {
    // Refines i < array.length to i < array.length - 1
    if (i != array.length - 1) {
      refineNeqLengthMTwo(array, i);
      // :: error: (argument)
      refineNeqLengthMThree(array, i);
    }
  }

  void refineNeqLengthMTwo(int[] array, @NonNegative @LTOMLengthOf("#1") int i) {
    // Refines i < array.length - 1 to i < array.length - 2
    if (i != array.length - 2) {
      refineNeqLengthMThree(array, i);
    }
    // No refinement
    if (i != array.length - 1) {
      // :: error: (argument)
      refineNeqLengthMThree(array, i);
    }
  }

  void refineNeqLengthMTwoNonLiteral(
      int[] array,
      @NonNegative @LTOMLengthOf("#1") int i,
      @IntVal(3) int c3,
      @IntVal({2, 3}) int c23) {
    // Refines i < array.length - 1 to i < array.length - 2
    if (i != array.length - (5 - c3)) {
      refineNeqLengthMThree(array, i);
    }
    // No refinement
    if (i != array.length - c23) {
      // :: error: (argument)
      refineNeqLengthMThree(array, i);
    }
  }

  @LTLengthOf(value = "#1", offset = "3") int refineNeqLengthMThree(
      int[] array, @NonNegative @LTLengthOf(value = "#1", offset = "2") int i) {
    // Refines i < array.length - 2 to i < array.length - 3
    if (i != array.length - 3) {
      return i;
    }
    // :: error: (return)
    return i;
  }

  // The same test for a string.
  @LTLengthOf(value = "#1", offset = "3") int refineNeqLengthMThree(
      String str, @NonNegative @LTLengthOf(value = "#1", offset = "2") int i) {
    // Refines i < str.length() - 2 to i < str.length() - 3
    if (i != str.length() - 3) {
      return i;
    }
    // :: error: (return)
    return i;
      private float __cfwr_util510(byte __cfwr_p0) {
        try {
            if (false || (null << -53.10f)) {
            double __cfwr_data74 = -70.17;
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        try {
            float __cfwr_result83 = ((386 * true) % (-651 + 78.91));
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        for (int __cfwr_i94 = 0; __cfwr_i94 < 2; __cfwr_i94++) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 3; __cfwr_i44++) {
            Double __cfwr_var59 = null;
        }
        }
        return 70.13f;
    }
    static float __cfwr_temp713() {
        Float __cfwr_entry62 = null;
        return null;
        return -28.28f;
    }
    private static Character __cfwr_compute116(Boolean __cfwr_p0, long __cfwr_p1, double __cfwr_p2) {
        return null;
        return null;
        return null;
        return null;
    }
}
