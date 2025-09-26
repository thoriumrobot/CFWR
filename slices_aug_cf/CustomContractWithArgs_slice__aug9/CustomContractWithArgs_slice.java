/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        Object __cfwr_item12 = null;

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
        public double __cfwr_helper588(short __cfwr_p0) {
        if (false && false) {
            try {
            while ((null * null)) {
            try {
            Integer __cfwr_var19 = null;
        } catch (Exception __cfwr_e11) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        while (false) {
            while (false) {
            Long __cfwr_node30 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 8; __cfwr_i43++) {
            while (false) {
            if (true || true) {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 8; __cfwr_i28++) {
            while ((-38.17 % (null & null))) {
            while (((616 << 815L) >> (null ^ 677))) {
            if (true || ('Q' << 450)) {
            try {
            while ((-58.76 + 49)) {
            try {
            if ((-561L ^ true) && false) {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 7; __cfwr_i72++) {
            if (('v' ^ -246) && (null * 748)) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 9; __cfwr_i8++) {
            Character __cfwr_item60 = null;
        }
        }
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        for (int __cfwr_i61 = 0; __cfwr_i61 < 1; __cfwr_i61++) {
            short __cfwr_entry34 = null;
        }
        return 67.76;
    }
    protected Double __cfwr_process270(byte __cfwr_p0, Float __cfwr_p1) {
        boolean __cfwr_val93 = true;
        return null;
    }
}
