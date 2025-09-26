/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        return "item64";

    // Refines i <= array.length to i < array.length
  
        try {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 4; __cfwr_i68++) {
            return ((null << -357L) << (-96.87 ^ 73.12));
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
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
      protected char __cfwr_calc747(Boolean __cfwr_p0, short __cfwr_p1, short __cfwr_p2) {
        return null;
        Boolean __cfwr_var3 = null;
        if ((864 | false) || false) {
            try {
            if (false && ((null - 639) * -72.02)) {
            while (false) {
            if ((true + -808) && false) {
            if (true && true) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 2; __cfwr_i40++) {
            if (false && false) {
            try {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 7; __cfwr_i29++) {
            return (false << (false + false));
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        return 'L';
    }
    protected static double __cfwr_util160() {
        if (true || true) {
            return 'Q';
        }
        while (((-525 | -843L) / false)) {
            while ((null & 636)) {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 4; __cfwr_i36++) {
            try {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 4; __cfwr_i7++) {
            while (false) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 9; __cfwr_i3++) {
            for (int __cfwr_i76 = 0; __cfwr_i76 < 3; __cfwr_i76++) {
            try {
            while ((-27.95 % '0')) {
            while (false) {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 10; __cfwr_i25++) {
            long __cfwr_result35 = -409L;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i40 = 0; __cfwr_i40 < 4; __cfwr_i40++) {
            try {
            return null;
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        }
        return 41.00;
    }
    protected static int __cfwr_process19() {
        if (true || false) {
            return null;
        }
        try {
            return null;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return -265;
    }
}
