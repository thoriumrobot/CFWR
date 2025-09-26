/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        try {
            while (false) {
            while (false) {
            while (true) {
            for (int __cfwr_i39 = 0; __cfwr_i39 < 5; __cfwr_i39++) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 4; __cfwr_i3++) {
            if (true && (0L << (-222L & 'v'))) {
            return true;
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e59) {
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
        protected static Long __cfwr_util30(String __cfwr_p0) {
        while (true) {
            try {
            try {
            String __cfwr_node85 = "world23";
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        } catch (Exception __cfwr_e63) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (false || (889L - -150L)) {
            if (true && (true << 'R')) {
            if (true && (false - (null % -945L))) {
            Object __cfwr_temp22 = null;
        }
        }
        }
        while (true) {
            while (true) {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 2; __cfwr_i25++) {
            byte __cfwr_node57 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        byte __cfwr_data82 = ((null & -680L) | null);
        return null;
    }
    protected static double __cfwr_calc996(Integer __cfwr_p0) {
        if (false || false) {
            if (true && false) {
            try {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 2; __cfwr_i55++) {
            while (true) {
            while (true) {
            Object __cfwr_node16 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        }
        }
        if (true || false) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 8; __cfwr_i90++) {
            while (('l' + ('5' ^ false))) {
            while (true) {
            Integer __cfwr_item56 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        Character __cfwr_result69 = null;
        return -56.82;
    }
}
