/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        Integer __cfwr_temp39 = null;

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
      Character __cfwr_calc728(Character __cfwr_p0, char __cfwr_p1, double __cfwr_p2) {
        if (((null ^ 112L) % false) && false) {
            try {
            if (true && (null + false)) {
            short __cfwr_obj95 = null;
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        }
        return null;
    }
    static boolean __cfwr_aux834(long __cfwr_p0) {
        try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 9; __cfwr_i77++) {
            if (false || true) {
            if (false && true) {
            return (15.49f & (null | null));
        }
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        Double __cfwr_node96 = null;
        for (int __cfwr_i59 = 0; __cfwr_i59 < 8; __cfwr_i59++) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 1; __cfwr_i75++) {
            return -8.31;
        }
        }
        return false;
    }
    private byte __cfwr_calc845(Character __cfwr_p0) {
        for (int __cfwr_i37 = 0; __cfwr_i37 < 6; __cfwr_i37++) {
            try {
            try {
            if (false || true) {
            float __cfwr_elem3 = ((-39.63f | 75.32) * (true & 281L));
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        }
        return null;
        float __cfwr_val28 = -73.46f;
        return (673 ^ (-23.39 | null));
    }
}
