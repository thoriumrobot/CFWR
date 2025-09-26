/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        if (false || false) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 1
        for (int __cfwr_i37 = 0; __cfwr_i37 < 4; __cfwr_i37++) {
            return null;
        }
0; __cfwr_i33++) {
            return (-724L ^ null);
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
      static boolean __cfwr_process404(Boolean __cfwr_p0, char __cfwr_p1, Character __cfwr_p2) {
        if (((null % -79.47f) | 6.87f) || (false % (null - -731))) {
            try {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 5; __cfwr_i41++) {
            if (((null * 15.93f) << null) || ((-4.78 + 'M') >> (55.60f << -21.81))) {
            if ((('j' ^ null) | true) && true) {
            while ((null ^ null)) {
            Boolean __cfwr_item59 = null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
        for (int __cfwr_i55 = 0; __cfwr_i55 < 3; __cfwr_i55++) {
            if (false || true) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        }
        }
        return true;
    }
}
