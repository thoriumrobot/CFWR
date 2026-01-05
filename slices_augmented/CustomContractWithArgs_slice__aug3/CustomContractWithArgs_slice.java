/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CustomContractWithArgs_slice {
    @Positive
    void minLenUse(int[] b) {
        while ((76.41 & null)) {
            try {
            Long __cfwr_temp52 = null;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
            break; // Prevent infinite loops
        }

    @Positive
      minLenContract(b);
    @Positive
      int @MinLen(10) [] c = b;
    @Positive
    }

    @Positive
    public int b, y;

    @Positive
        expression = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#2 + 1", "10"},
    @Positive
        result = true)
    @Positive
    boolean ltlPost(int[] a, int c) {
    @Positive
      if (b < a.length - c - 1 && b < a.length - 10) {
    @Positive
        return true;
    @Positive
      } else {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }

    // :: error: (flowexpr.parse.error)
    @Positive
    boolean ltlPostInvalid(int[] a, int c) {
    @Positive
      return false;
    @Positive
    }

    @Positive
        value = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#2 + 1", "-10"})
    @Positive
    void ltlPre(int[] a, int c) {
    @Positive
      @LTLengthOf(value = "a", offset = "c+1") int i = b;
    @Positive
    }

    @Positive
    void ltlUse(int[] a, int c) {
    @Positive
      if (ltlPost(a, c)) {
    @Positive
        @LTLengthOf(value = "a", offset = "c+1") int i = b;

    @Positive
        ltlPre(a, c);
    @Positive
      }
      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "c+1") int j = b;
    @Positive
    }

    protected static Object __cfwr_aux851(int __cfwr_p0, double __cfwr_p1, Character __cfwr_p2) {
        try {
            if (false && ((null - '0') % 'z')) {
            try {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 7; __cfwr_i36++) {
            Character __cfwr_var84 = null;
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        return null;
    }
    protected Boolean __cfwr_calc121() {
        if (true || (null % true)) {
            if (true || true) {
            while (true) {
            if (false || (null * -84L)) {
            float __cfwr_var84 = (-36.52f % (-58.89 * null));
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}