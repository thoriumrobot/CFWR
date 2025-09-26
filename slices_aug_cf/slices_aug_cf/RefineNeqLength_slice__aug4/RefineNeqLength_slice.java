/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        for (int __cfwr_i23 = 0; __cfwr_i23 < 1; __cfwr_i23++) {
            whil
        short __cfwr_item26 = null;
e (true) {
            Boolean __cfwr_data18 = null;
            break; // Prevent infinite loops
        }
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
      private Float __cfwr_aux60(double __cfwr_p0) {
        Boolean __cfwr_entry78 = null;
        return (-290 * 56.60);
        return "test96";
        for (int __cfwr_i94 = 0; __cfwr_i94 < 2; __cfwr_i94++) {
            Integer __cfwr_var35 = null;
        }
        return null;
    }
    private Boolean __cfwr_process369() {
        try {
            if (false && true) {
            if (true && true) {
            if (((null + 44.06f) & ('5' - -543L)) && true) {
            while (false) {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 9; __cfwr_i12++) {
            while ((true / false)) {
            if (true && (302 | 75.44f)) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 7; __cfwr_i1++) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        if (true && true) {
            short __cfwr_node53 = null;
        }
        for (int __cfwr_i59 = 0; __cfwr_i59 < 2; __cfwr_i59++) {
            try {
            boolean __cfwr_var10 = true;
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        }
        return ('V' % null);
        return null;
    }
    public static byte __cfwr_temp799() {
        char __cfwr_elem60 = 'h';
        try {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 4; __cfwr_i99++) {
            while (false) {
            double __cfwr_item88 = 40.94;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        return (93.68 & -900L);
    }
}
