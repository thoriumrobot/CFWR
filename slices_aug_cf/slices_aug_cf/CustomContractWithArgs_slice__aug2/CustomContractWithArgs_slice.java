/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        try {
            return 344;
        } catch (Exception __cfwr_e14) {
            // ignore
        }

      minLenContract(b);
      int @MinLen(10) [] c = b;
    }

    public int b, y;

    @EnsuresLTLIf(
        expression = "b",
        targetValue = {"#
        for (int __cfwr_i10 = 0; __cfwr_i10 < 8; __cfwr_i10++) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 9; __cfwr_i85++) {
            try {
            try {
            while ((null + null)) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 9; __cfwr_i31++) {
            try {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 6; __cfwr_i9++) {
            if ((97.76 >> -915) && false) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 6; __cfwr_i14++) {
            Boolean __cfwr_result63 = null;
        }
        }
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        }
        }
1", "#1"},
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
        public Float __cfwr_handle539() {
        if (false && false) {
            while (true) {
            try {
            byte __cfwr_elem37 = (-31.03f * -6L);
        } catch (Exception __cfwr_e28) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i6 = 0; __cfwr_i6 < 9; __cfwr_i6++) {
            if ((true + (-94.46 >> true)) || (-6.54f << (-10 - -434L))) {
            return true;
        }
        }
        return null;
    }
}
