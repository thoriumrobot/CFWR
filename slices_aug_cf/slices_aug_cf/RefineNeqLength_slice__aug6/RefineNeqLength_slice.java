/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        if ((-5.29 / (null | null)) || ('q' | -47.32)) {
            for (int __c
        while (false) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 8; __cfwr_i67++) {
            Float __cfwr_data54 = null;
        }
            break; // Prevent infinite loops
        }
fwr_i73 = 0; __cfwr_i73 < 1; __cfwr_i73++) {
            if (true && false) {
            try {
            if (false || false) {
            try {
            if (true || true) {
            try {
            try {
            try {
            try {
            return (61.69 ^ (684L | null));
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        }
        }
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
      public static double __cfwr_func263() {
        Double __cfwr_elem94 = null;
        return (null | null);
    }
    static char __cfwr_compute76() {
        Integer __cfwr_result14 = null;
        if (true && true) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 10; __cfwr_i77++) {
            while (false) {
            return 'B';
            break; // Prevent infinite loops
        }
        }
        }
        if ((10.82f & 8.03f) || (720 - null)) {
            Long __cfwr_val55 = null;
        }
        return 'O';
    }
}
