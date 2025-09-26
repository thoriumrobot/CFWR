/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        Boolean __cfwr_temp81 = null;

      minLenContract(b);
      int @MinLen(10) [] c = b;
    }

    public int b, y;

    @EnsuresLTLIf(
        expression = "b",
        targetValue = {"#1", "#1"},
        targetOffset = {"#2 + 1", "10"},
        result = true
        Long __cfwr_result80 = null;
)
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
        protected static Float __cfwr_compute856(Double __cfwr_p0, boolean __cfwr_p1) {
        try {
            try {
            if (true && (153 / 141L)) {
            Long __cfwr_obj8 = null;
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        return null;
    }
    protected Object __cfwr_util537(Boolean __cfwr_p0, String __cfwr_p1, String __cfwr_p2) {
        try {
            try {
            while ((null >> null)) {
            while (('K' & (null / true))) {
            boolean __cfwr_obj43 = false;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        for (int __cfwr_i58 = 0; __cfwr_i58 < 2; __cfwr_i58++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 9; __cfwr_i61++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 2; __cfwr_i61++) {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 8; __cfwr_i58++) {
            return null;
        }
        }
        }
        }
        return null;
    }
    public static long __cfwr_calc917() {
        float __cfwr_obj21 = (null ^ (null >> -3.64));
        if (false || true) {
            boolean __cfwr_entry84 = false;
        }
        for (int __cfwr_i57 = 0; __cfwr_i57 < 10; __cfwr_i57++) {
            return null;
        }
        try {
            Boolean __cfwr_obj29 = null;
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        return 486L;
    }
}
