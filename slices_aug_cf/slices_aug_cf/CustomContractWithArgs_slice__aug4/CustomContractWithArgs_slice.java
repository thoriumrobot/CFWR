/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        try {
            if ((21.74f % 87.95) || true) {
            for (int __cfwr_i86 = 0; __cfwr_i86 < 9; __cfwr_i86++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }

      minLenContract(b
        if (((32.27 & 10.42f) | null) || (true / (true * -17.29))) {
            while ((483 << (null - null))) {
            try {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 1; __cfwr_i38++) {
            char __cfwr_item2 = '4';
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
);
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
        static String __cfwr_util57(Integer __cfwr_p0, Double __cfwr_p1) {
        return ((true ^ 86.29) ^ ('J' ^ null));
        while (true) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 7; __cfwr_i29++) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 10; __cfwr_i69++) {
            if (true || true) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        if (true && false) {
            char __cfwr_temp32 = (-649L % (-95.03f - 86.09));
        }
        if (true || (null >> (-89.58f * -69.42f))) {
            if (((-733 + null) & -379L) || ((770 | -7.28) | 21.37)) {
            String __cfwr_temp6 = "test3";
        }
        }
        return "test99";
    }
    static Boolean __cfwr_process170(char __cfwr_p0, Integer __cfwr_p1, byte __cfwr_p2) {
        return null;
        if (true && false) {
            try {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 10; __cfwr_i97++) {
            Long __cfwr_var38 = null;
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        }
        return null;
    }
}
