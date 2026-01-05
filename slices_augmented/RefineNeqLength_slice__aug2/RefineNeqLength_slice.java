/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeqLength_slice {
    @Positive
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        try {
            return null;
        } catch (Exception __cfwr_e5) {
            // ignore
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

    public static Integer __cfwr_util423(Float __cfwr_p0, long __cfwr_p1, float __cfwr_p2) {
        if (false || false) {
            return -219L;
        }
        if (false && false) {
            try {
            for (int __cfwr_i39 = 0; __cfwr_i39 < 5; __cfwr_i39++) {
            if (((null << -7.70f) * -39.96) && (-222L - 292)) {
            try {
            String __cfwr_data78 = "result1";
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        }
        return null;
    }
    static float __cfwr_util835(Integer __cfwr_p0) {
        return false;
        while (true) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 1; __cfwr_i9++) {
            if ((56L % (-300L << true)) && true) {
            while (true) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 8; __cfwr_i40++) {
            Long __cfwr_var1 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i7 = 0; __cfwr_i7 < 7; __cfwr_i7++) {
            while (((-276L / false) << (32.28 ^ -46.24))) {
            Boolean __cfwr_result4 = null;
            break; // Prevent infinite loops
        }
        }
        while ((null >> (985L >> -42.21f))) {
            if (false || (('S' ^ -90.91f) | 89.59f)) {
            if ((234L + -320) || true) {
            if (false && false) {
            long __cfwr_result57 = (null << null);
        }
        }
        }
            break; // Prevent infinite loops
        }
        return 27.12f;
    }
    protected double __cfwr_compute13(Double __cfwr_p0) {
        long __cfwr_elem60 = (null - null);
        return ('Z' ^ (null + -95L));
    }
}