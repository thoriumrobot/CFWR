/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        for (int __cfwr_i29 = 0; __cfwr_i29 < 9; __cfwr_i29++) {
            while ((-675L ^ 'l')) {
            return 355L;
            break; // Prevent infinite loops
        }
        }

      minLenContract(b);
      int @MinLen(10) [] c = b;
    }

    public i
        if (true || (null & 'z')) {
            if ((false * null) || true) {
            if (true && (92.64f % -876L)) {
            try {
            while (false) {
            while (false) {
            return (-936L % 62.43f);
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        }
        }
        }
nt b, y;

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
        Long __cfwr_util161(int __cfwr_p0, Integer __cfwr_p1, Character __cfwr_p2) {
        return 'O';
        return null;
    }
    public static long __cfwr_helper9() {
        try {
            double __cfwr_data8 = -89.89;
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        try {
            if ((-33.15f + null) && true) {
            while (true) {
            Double __cfwr_item95 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        return 889L;
    }
}
