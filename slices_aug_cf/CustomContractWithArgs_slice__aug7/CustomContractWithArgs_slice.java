/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        return null;

      minLenContract(b);
      int @MinLen(10) [] c = b;
    }

    public int b, y;

    @EnsuresLTLIf(
        expression = "b",
        targetValue = {"#1", "#1"},
        targetOffset = {"#2 + 1", "10"},
        result = true)
    boolean ltl
        while ((-650L % (null & -47.63f))) {
            try {
            Integer __cfwr_obj70 = null;
        } catch (Exception __cfwr_e70) {
            // ignore
        }
            break; // Prevent infinite loops
        }
Post(int[] a, int c) {
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
        protected double __cfwr_compute535(byte __cfwr_p0, char __cfwr_p1) {
        if (false && true) {
            char __cfwr_entry3 = 'w';
        }
        while ((-380L + 224L)) {
            long __cfwr_result58 = 420L;
            break; // Prevent infinite loops
        }
        return 93.38;
    }
    static long __cfwr_helper548(float __cfwr_p0) {
        while ((25.30f | 560L)) {
            return null;
            break; // Prevent infinite loops
        }
        try {
            try {
            long __cfwr_entry8 = 253L;
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        try {
            return null;
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        return -545L;
    }
    protected byte __cfwr_aux698(char __cfwr_p0) {
        while (true) {
            if (false || (-474 % '1')) {
            if (((null + 23.28f) % (-91.12f & null)) && (null + null)) {
            while (false) {
            if (false || (true ^ (84.40f | false))) {
            while (true) {
            try {
            if (false && false) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 10; __cfwr_i59++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        try {
            long __cfwr_temp67 = -87L;
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        return null;
    }
}
