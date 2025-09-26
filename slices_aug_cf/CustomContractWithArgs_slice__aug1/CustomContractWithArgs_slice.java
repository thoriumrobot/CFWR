/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    void minLenUse(int[] b) {
        while (((true + null) << null)) {
            while (((false / -955) + (true << 316))) {
            try {
            return (true * '1');
        } catch (Exception __cfwr_e96) {
            // ignore
        }
            break; // Prevent infinite loops
  
        while (false) {
            for (int __cfwr_i37 = 0; __cfwr_i37 < 1; __cfwr_i37++) {
            while (true) {
            while (true) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 3; __cfwr_i56++) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 1; __cfwr_i3++) {
            if (false || true) {
            while (true) {
            return 628;
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
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
        private static Integer __cfwr_compute843(Double __cfwr_p0, Float __cfwr_p1) {
        if (false || true) {
            try {
            if (false || ('y' % null)) {
            boolean __cfwr_temp86 = false;
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        }
        try {
            return null;
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        while (true) {
            while (true) {
            while (true) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        boolean __cfwr_obj66 = true;
        return null;
    }
    public short __cfwr_util370(long __cfwr_p0, Character __cfwr_p1) {
        Double __cfwr_data32 = null;
        return null;
        if (false && true) {
            while (true) {
            int __cfwr_item64 = 960;
            break; // Prevent infinite loops
        }
        }
        Boolean __cfwr_node11 = null;
        return ((-767 * -179) << -878);
    }
    protected static int __cfwr_process409() {
        if ((-930L - (-1.24f + -985L)) || true) {
            try {
            try {
            return true;
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        }
        byte __cfwr_elem60 = (-675 ^ null);
        while (false) {
            return -501;
            break; // Prevent infinite loops
        }
        return null;
        return -903;
    }
}
