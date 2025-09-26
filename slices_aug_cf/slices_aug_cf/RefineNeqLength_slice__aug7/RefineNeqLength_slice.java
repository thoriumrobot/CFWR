/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
        for (int __cfwr_i27 = 0; __cfwr_i27 < 3; __cfwr_i27++) {
            try 
        try {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 3; __cfwr_i61++) {
            if (true || true) {
            return "test54";
        }
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
{
            if (true && false) {
            while (false) {
            while ((78.40f << 'H')) {
            if ((-31.13f + (true | -41.83)) || true) {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 6; __cfwr_i63++) {
            if (((-66.20f * 332) | -24.96f) || true) {
            try {
            float __cfwr_val7 = -16.64f;
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e16) {
            // ignore
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
      private static float __cfwr_proc889(Float __cfwr_p0, String __cfwr_p1) {
        return null;
        while ((null % -19.66)) {
            while (false) {
            return ((true % true) ^ ('M' ^ true));
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        while ((null | 'k')) {
            Object __cfwr_node52 = null;
            break; // Prevent infinite loops
        }
        try {
            try {
            char __cfwr_obj26 = 'p';
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        return (-7.16f & true);
    }
    private static Long __cfwr_compute609(char __cfwr_p0, Character __cfwr_p1, Long __cfwr_p2) {
        if (false && (597 | 71.42)) {
            return null;
        }
        for (int __cfwr_i84 = 0; __cfwr_i84 < 7; __cfwr_i84++) {
            try {
            int __cfwr_elem17 = 231;
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        }
        if ((false % null) && false) {
            try {
            int __cfwr_node85 = 719;
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        }
        return null;
    }
    protected float __cfwr_compute66(String __cfwr_p0, Integer __cfwr_p1) {
        for (int __cfwr_i84 = 0; __cfwr_i84 < 5; __cfwr_i84++) {
            if ((394 / (null + false)) && true) {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 4; __cfwr_i93++) {
            return null;
        }
        }
        }
        return null;
        if (true || (false * -55.55)) {
            if (false || false) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 10; __cfwr_i34++) {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 2; __cfwr_i60++) {
            while ((24L >> (true | null))) {
            if (false && true) {
            Character __cfwr_temp81 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        return -88.40f;
    }
}
