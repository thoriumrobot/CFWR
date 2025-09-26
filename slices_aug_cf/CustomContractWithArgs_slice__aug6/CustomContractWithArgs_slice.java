/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        while (true) {
            while (false) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 3; __cfwr_i67++) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 6; __cfwr_i91++) {
            if (true || true) {
            try {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 8; __cfwr_i5++) {
            if ((-39.10 + (30.46 * 670)) || false) {
            try {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 8; __cfwr_i71++) {
            char __cfwr_item65 = (null | 65.58f);
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
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
        protected float __cfwr_proc497(Integer __cfwr_p0) {
        for (int __cfwr_i77 = 0; __cfwr_i77 < 3; __cfwr_i77++) {
            return -148L;
        }
        return null;
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return ((-35 & 7.95) ^ (null * 'S'));
    }
}
