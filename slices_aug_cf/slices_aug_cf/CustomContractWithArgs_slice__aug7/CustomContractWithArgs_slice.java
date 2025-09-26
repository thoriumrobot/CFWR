/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        for (int __cfwr_i38 = 0; __cfwr_i38 < 8; __cfwr_i38++) {
            long __cfwr_temp60 = 971L;
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
        private static Object __cfwr_proc887(Float __cfwr_p0, long __cfwr_p1, Character __cfwr_p2) {
        try {
            try {
            try {
            try {
            try {
            while (true) {
            try {
            while (false) {
            if (('Y' >> -88.31) || false) {
            if (false && true) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 10; __cfwr_i29++) {
            return null;
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        if (('e' << null) && true) {
            try {
            if (false && ((null & -62.74) >> 31.61f)) {
            return null;
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        }
        try {
            return "world16";
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        float __cfwr_val63 = (68.19 & 'z');
        return null;
    }
    protected static Long __cfwr_func114() {
        return -602;
        return null;
    }
}
