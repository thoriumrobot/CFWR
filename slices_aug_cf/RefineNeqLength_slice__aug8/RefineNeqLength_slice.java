/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        if ((true / -698) || true) {
            byte __cfwr_result3 = (null - 49
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
.19f);
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
      public Boolean __cfwr_handle21(Object __cfwr_p0) {
        if (true && ('k' / null)) {
            try {
            return null;
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        }
        if (true && (null & -75.31f)) {
            int __cfwr_temp48 = 251;
        }
        for (int __cfwr_i38 = 0; __cfwr_i38 < 5; __cfwr_i38++) {
            return null;
        }
        while (true) {
            if (false || true) {
            float __cfwr_elem67 = (62.09 & null);
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
