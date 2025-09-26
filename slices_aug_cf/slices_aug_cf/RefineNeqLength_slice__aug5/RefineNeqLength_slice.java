/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        while (false) {
            if (((-71.87f & -513L) + -346) && true) {
            return null;
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
      private double __cfwr_helper67(Boolean __cfwr_p0) {
        Integer __cfwr_data6 = null;
        char __cfwr_entry56 = 'z';
        return -23.45;
    }
    static Long __cfwr_aux931(String __cfwr_p0) {
        while (true) {
            try {
            try {
            try {
            if (false || false) {
            double __cfwr_result50 = (56.38f / 115);
        }
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        try {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 5; __cfwr_i50++) {
            if ((55.71f >> ('g' ^ -51.77f)) && true) {
            try {
            while (true) {
            if (('w' & 154) && true) {
            while (true) {
            while (((-588 ^ 'v') + 83.76f)) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 1; __cfwr_i8++) {
            if (true && false) {
            Long __cfwr_var76 = null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        if (false || true) {
            return null;
        }
        return null;
    }
}
