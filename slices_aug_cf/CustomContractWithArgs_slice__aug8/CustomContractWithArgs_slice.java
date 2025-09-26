/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        for (int __cfwr_i98 = 0; __cfwr_i98 < 4; __cfwr_i98++) {
            while (('w' | (-98.22f / 818))) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 5; __cfwr_i34++) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 3; __cfwr_i20++) {
            byte __cfwr_data63 = null;
        }
        }
            break; // Prevent infinite loops
        }
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
        private static float __cfwr_aux345(float __cfwr_p0, Boolean __cfwr_p1) {
        if (((null & -578) & (null / false)) && false) {
            double __cfwr_item98 = 76.94;
        }
        return null;
        return -53.26f;
    }
    protected static Character __cfwr_compute964(long __cfwr_p0) {
        return null;
        try {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 8; __cfwr_i97++) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 3; __cfwr_i48++) {
            if (((null + -77.24) << null) || false) {
            return null;
        }
        }
        }
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        return null;
    }
    private static byte __cfwr_proc894(boolean __cfwr_p0) {
        if (true || false) {
            return null;
        }
        while (true) {
            if (false && true) {
            if (true || false) {
            while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e8) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
