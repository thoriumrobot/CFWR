/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeqLength_slice {
    @Positive
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        try {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 3; __cfwr_i20++) {
            
        Object __cfwr_item70 = null;
Character __cfwr_item1 = null;
        }
        } catch (Exception __cfwr_e58) {
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

    private Character __cfwr_compute88(Integer __cfwr_p0, Long __cfwr_p1) {
        return null;
        return null;
    }
    long __cfwr_temp945(float __cfwr_p0, boolean __cfwr_p1) {
        return "data60";
        return (-95.50 | (true & -346L));
    }
    protected Boolean __cfwr_aux739(byte __cfwr_p0, Long __cfwr_p1, Character __cfwr_p2) {
        for (int __cfwr_i9 = 0; __cfwr_i9 < 10; __cfwr_i9++) {
            return ((-8.79f ^ null) ^ -697L);
        }
        for (int __cfwr_i51 = 0; __cfwr_i51 < 8; __cfwr_i51++) {
            while (('k' * (958 + 521))) {
            try {
            return true;
        } catch (Exception __cfwr_e81) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i82 = 0; __cfwr_i82 < 9; __cfwr_i82++) {
            if (true && true) {
            while (true) {
            char __cfwr_elem30 = 'w';
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}