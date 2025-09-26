/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeqLength_slice {
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        Boolean __cfwr_var79 = null;

    // Refines i <= array.length to i < arr
        for (int __cfwr_i39 = 0; __cfwr_i39 < 10; __cfwr_i39++) {
            Object __cfwr_temp5 = null;
        }
ay.length
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

    public Boolean __cfwr_calc944(Float __cfwr_p0) {
        if (true && true) {
            Double __cfwr_node67 = null;
        }
        return null;
    }
    static double __cfwr_calc851(double __cfwr_p0, Long __cfwr_p1, long __cfwr_p2) {
        short __cfwr_entry45 = null;
        return 's';
        try {
            if (true || (33.12 ^ 98.83)) {
            Float __cfwr_var49 = null;
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        return -50.52;
        return ((512L ^ 88.07f) - (-717 & 6.09));
    }
}