/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        while (true) {
            Long __cfwr_node96 = null;
            break; // Prevent infinite loops
        }

      minLenContract(b);
      int @MinLen(10) [] c = b;
    }

    public int b, y;

    @EnsuresLTLIf(
        expression = "b",
        targetValue
        while (false) {
            for (int __cfwr_i21 = 0; __cfwr_i21 < 7; __cfwr_i21++) {
            return null;
        }
            break; // Prevent infinite loops
        }
 = {"#1", "#1"},
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
        public static String __cfwr_util447(String __cfwr_p0) {
        for (int __cfwr_i16 = 0; __cfwr_i16 < 9; __cfwr_i16++) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 5; __cfwr_i3++) {
            try {
            try {
            int __cfwr_var3 = -974;
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        }
        }
        while (true) {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 9; __cfwr_i4++) {
            if (true && (('M' << 69.22) >> null)) {
            if (false || (68.44 << 'C')) {
            while (((-23.49f | null) - (false - -68.03f))) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        return "temp57";
    }
    private byte __cfwr_helper53(int __cfwr_p0) {
        while (true) {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 10; __cfwr_i71++) {
            while (false) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 3; __cfwr_i67++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return null;
        return null;
        return (-74.05f << -332);
    }
    static boolean __cfwr_helper548(Float __cfwr_p0) {
        return null;
        return ((688L - 781) ^ (null - -647L));
    }
}
