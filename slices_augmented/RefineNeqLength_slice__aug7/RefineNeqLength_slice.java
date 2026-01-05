/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeqLength_slice {
    @Positive
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        for (int __cfwr_i84 = 0; __cfwr_i84 < 4; __cfwr_i84++) {
            for (int __cfwr_i9
        return "data79";
9 = 0; __cfwr_i99 < 8; __cfwr_i99++) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 2; __cfwr_i73++) {
            return null;
        }
        }
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

    static Object __cfwr_temp183(Long __cfwr_p0) {
        return null;
        for (int __cfwr_i77 = 0; __cfwr_i77 < 3; __cfwr_i77++) {
            Integer __cfwr_temp73 = null;
        }
        for (int __cfwr_i57 = 0; __cfwr_i57 < 8; __cfwr_i57++) {
            try {
            try {
            float __cfwr_obj58 = 59.50f;
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        }
        return null;
    }
    protected Float __cfwr_temp453(Integer __cfwr_p0, float __cfwr_p1) {
        while (('c' >> null)) {
            Long __cfwr_item24 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
    long __cfwr_helper938(String __cfwr_p0) {
        while (true) {
            if (true && false) {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 2; __cfwr_i50++) {
            return 46.10f;
        }
        }
            break; // Prevent infinite loops
        }
        try {
            try {
            return null;
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        } catch (Exception __cfwr_e17) {
            // ignore
        }
        while ((null & -549)) {
            if (false || true) {
            try {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 9; __cfwr_i94++) {
            try {
            try {
            while (true) {
            try {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 10; __cfwr_i31++) {
            while (true) {
            int __cfwr_val61 = (false >> false);
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return 896L;
    }
}