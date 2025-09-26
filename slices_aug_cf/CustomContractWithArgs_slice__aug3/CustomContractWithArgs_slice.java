/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        try {
            if (false || true) {
            while (true) {
            try {
            return ((null | -987) | (null - null));
        } catch (Exception __cfwr_e10) {
            // ignore
        }
            break; // Prevent infinite loops
      
        if (false && ('j' + (null | -76.50f))) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        }
  }
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }

      minLenContract(b);
      int @MinLen(10) [] c = b;
    }

    public int b, y;

    @EnsuresLTLIf(
        expression = "b",
        targetValue = {"#1", "#1"},
        targetOffset = {"#2 + 1", "10"},
        result = true)
    boolean ltlPost(int[] a, int c) {
      if (b < a.length - c - 1 && b < a.length - 10) {
        return true;
      } else {
        return false;
      }
    }

    @EnsuresLTLIf(expression = "b", targetValue = "#1", targetOffset = "#3", result = true)
    // :: error: (flowexpr.parse.error)
    boolean ltlPostInvalid(int[] a, int c) {
      return false;
    }

    @RequiresLTL(
        value = "b",
        targetValue = {"#1", "#1"},
        targetOffset = {"#2 + 1", "-10"})
    void ltlPre(int[] a, int c) {
      @LTLengthOf(value = "a", offset = "c+1") int i = b;
    }

    void ltlUse(int[] a, int c) {
      if (ltlPost(a, c)) {
        @LTLengthOf(value = "a", offset = "c+1") int i = b;

        ltlPre(a, c);
      }
      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "c+1") int j = b;
    }
  }

  class Derived extends Base {
    public int x;

    @Override
    @EnsuresLTLIf(
        expression = "b ",
        targetValue = {"#1", "#1"},
        targetOffset = {"#2 + 1", "11"},
        result = true)
    boolean ltlPost(int[] a, int d) {
      return false;
    }

    @Override
    @RequiresLTL(
        value = "b ",
        targetValue = {"#1", "#1"},
        targetOffset = {"#2 + 1", "-11"})
    void ltlPre(int[] a, int d) {
      @LTLengthOf(
          value = {"a", "a"},
          offset = {"d+1", "-10"})
      // :: error: (assignment)
      int i = b;
    }
  }

  class DerivedInvalid extends Base {
    public int x;

    @Override
    @EnsuresLTLIf(
        expression = "b ",
        targetValue = {"#1", "#1"},
        targetOffset = {"#2 + 1", "9"},
        result = true)
    // :: error: (contracts.conditional.postcondition.true.override)
    boolean ltlPost(int[] a, int c) {
      // :: error: (contracts.conditional.postcondition)
      return true;
        private static long __cfwr_temp572() {
        if (true || true) {
            while (false) {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 6; __cfwr_i71++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        if (true || ((27.57f | 'd') & null)) {
            if ((6.63f >> null) && ((755 * -17.35) - null)) {
            while (false) {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 1; __cfwr_i28++) {
            try {
            if (true && false) {
            float __cfwr_temp88 = -92.07f;
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        return -337L;
        return -816L;
    }
    static short __cfwr_temp272(Double __cfwr_p0, short __cfwr_p1) {
        while (true) {
            if (true && ((-51.25 << false) | true)) {
            return 85.88;
        }
            break; // Prevent infinite loops
        }
        boolean __cfwr_entry29 = false;
        Double __cfwr_data79 = null;
        while (false) {
            while (((-83.70f << 'c') + -551L)) {
            if ((-85.53f | null) || true) {
            if ((-731 * (null % '1')) && true) {
            try {
            Float __cfwr_data91 = null;
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
