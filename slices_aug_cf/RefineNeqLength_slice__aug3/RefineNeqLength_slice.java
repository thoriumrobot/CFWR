/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        try {
            if (false || true) {
            byte __cfwr_obj53 = ((null % true) & (null * '1'));
        }
        } catch (Exception __cfwr_e3) {
            // ignore
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
      Character __cfwr_calc923() {
        if (true && true) {
            while (false) {
            if ((55.00f - null) && true) {
            try {
            if (((-86.82 & 54.14) >> '3') && false) {
            while (('F' + (null - null))) {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 3; __cfwr_i12++) {
            while (false) {
            double __cfwr_temp49 = (null + 602);
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        while ((-68.20f * -46.31f)) {
            try {
            return null;
        } catch (Exception __cfwr_e20) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (true || true) {
            if (false && true) {
            Long __cfwr_node90 = null;
        }
        }
        return null;
    }
}
