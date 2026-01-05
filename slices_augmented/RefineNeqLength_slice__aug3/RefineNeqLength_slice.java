/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeqLength_slice {
    @Positive
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        for (int __cfwr_i93 = 0; __cfwr_i93 < 7; __cfwr_i93++) {
            boolean __cfwr_ite
        if ((-752L / null) && true) {
            boolean __cfwr_obj53 = (null - (-111L >> null));
        }
m10 = true;
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

    private static double __cfwr_proc294(double __cfwr_p0, Long __cfwr_p1) {
        try {
            while (true) {
            if ((false << null) && (null - -263)) {
            if (true || true) {
            while (false) {
            if (true || false) {
            if ((38.89f ^ (null << 49.25)) || false) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 4; __cfwr_i85++) {
            try {
            if (false || false) {
            return 91.79;
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        if (false || true) {
            while ((689L & null)) {
            try {
            if ((null - 62.29) && false) {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 5; __cfwr_i51++) {
            while ((('2' % 'M') >> -487)) {
            return true;
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        try {
            while (true) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 3; __cfwr_i42++) {
            try {
            return 272L;
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        return -98.58;
    }
    private static long __cfwr_util838(Float __cfwr_p0, double __cfwr_p1) {
        Boolean __cfwr_obj72 = null;
        for (int __cfwr_i99 = 0; __cfwr_i99 < 10; __cfwr_i99++) {
            return ((-39.41 << 66.44f) - 565);
        }
        return null;
        if (true && (933L % null)) {
            return null;
        }
        return 845L;
    }
}