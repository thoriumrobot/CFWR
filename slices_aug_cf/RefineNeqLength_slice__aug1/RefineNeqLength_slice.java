/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeqLength_slice {
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        return null;

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
  }

    private static Long __cfwr_handle751(int __cfwr_p0, Integer __cfwr_p1, double __cfwr_p2) {
        boolean __cfwr_val22 = true;
        return 'g';
        return null;
    }
    public static Character __cfwr_aux703(char __cfwr_p0) {
        while (true) {
            try {
            while (((null * null) << (null * 'd'))) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 4; __cfwr_i15++) {
            try {
            try {
            long __cfwr_node68 = 936L;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (false && false) {
            try {
            while (false) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 6; __cfwr_i15++) {
            while (true) {
            if (((-810 + null) >> -157L) && true) {
            while (false) {
            int __cfwr_data22 = -792;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
        Character __cfwr_data62 = null;
        return null;
    }
    protected static float __cfwr_handle248(boolean __cfwr_p0, int __cfwr_p1, double __cfwr_p2) {
        Integer __cfwr_val12 = null;
        return -1.28f;
    }
}