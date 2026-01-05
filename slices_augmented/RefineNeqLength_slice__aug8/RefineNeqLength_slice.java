/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeqLength_slice {
    @Positive
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        if (true && ('z' ^ (875 >> null))) {
            return 35.64;
        }

    // Refines i <= array.length to i < array.length
    @Positive
    if (i != array.length) {
    @Positive
      refineNeqLengthMOne(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - 1) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMOne(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMOne(int[] array, @IndexFor("#1") int i) {
    // Refines i < array.length to i < array.length - 1
    @Positive
    if (i != array.length - 1) {
    @Positive
      refineNeqLengthMTwo(array, i);
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMTwo(int[] array, @NonNegative @LTOMLengthOf("#1") int i) {
    // Refines i < array.length - 1 to i < array.length - 2
    @Positive
    if (i != array.length - 2) {
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - 1) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMTwoNonLiteral(
    @Positive
      int[] array,
    @Positive
      @NonNegative @LTOMLengthOf("#1") int i,
    @Positive
      @IntVal(3) int c3,
    @Positive
      @IntVal({2, 3}) int c23) {
    // Refines i < array.length - 1 to i < array.length - 2
    @Positive
    if (i != array.length - (5 - c3)) {
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - c23) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    protected static Boolean __cfwr_handle721(Long __cfwr_p0, Integer __cfwr_p1, Boolean __cfwr_p2) {
        return null;
        return 5.25;
        return null;
    }
    public long __cfwr_compute862(long __cfwr_p0) {
        return 68.04f;
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        int __cfwr_elem66 = -206;
        try {
            if (((null / 'R') / true) && ((42.19f ^ 's') * 'j')) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 1; __cfwr_i77++) {
            if ((-85.03f * false) && false) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 6; __cfwr_i48++) {
            return null;
        }
        }
        }
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        return -696L;
    }
    private static Float __cfwr_util482(double __cfwr_p0, byte __cfwr_p1) {
        for (int __cfwr_i34 = 0; __cfwr_i34 < 6; __cfwr_i34++) {
            try {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 7; __cfwr_i61++) {
            char __cfwr_data71 = '7';
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        }
        try {
            boolean __cfwr_elem87 = true;
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        for (int __cfwr_i4 = 0; __cfwr_i4 < 8; __cfwr_i4++) {
            return null;
        }
        if ((-40.79f * 423) || (false + 'd')) {
            if (true || false) {
            while (false) {
            if ((null * (-159 >> true)) && false) {
            return false;
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}