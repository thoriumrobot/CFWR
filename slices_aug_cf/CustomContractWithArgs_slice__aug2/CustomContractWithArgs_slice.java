/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        for (int __cfwr_i72 = 0; __cfwr_i72 < 5; __cfwr_i72++) {
            try {
            while (true) {
            if (false || true) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e40) {
            // ignore
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
        public static String __cfwr_aux812(float __cfwr_p0) {
        try {
            String __cfwr_entry97 = "world7";
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return null;
        if ((-72 ^ '2') || (('j' + 'F') / '4')) {
            for (int __cfwr_i16 = 0; __cfwr_i16 < 4; __cfwr_i16++) {
            while ((('L' / null) / 779)) {
            try {
            return (762 >> (34.88f * null));
        } catch (Exception __cfwr_e61) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        return "test19";
    }
    public String __cfwr_helper705(short __cfwr_p0) {
        if (false || true) {
            while ((null ^ 90.12f)) {
            try {
            while ((71.57 / 'h')) {
            return (31.89f ^ 75.03);
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return "test58";
    }
}
