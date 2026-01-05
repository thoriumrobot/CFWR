/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CustomContractWithArgs_slice {
    @Positive
    void minLenUse(int[] b) {
        String __cfwr_entry24 = "world89";

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

    static Long __cfwr_handle500(byte __cfwr_p0) {
        if (true || true) {
            try {
            while (false) {
            while (false) {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 2; __cfwr_i63++) {
            return null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        }
        return null;
    }
    protected char __cfwr_aux846(Double __cfwr_p0, int __cfwr_p1, double __cfwr_p2) {
        try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 5; __cfwr_i32++) {
            if (false && (-706 >> null)) {
            return (-37.07 * null);
        }
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        return "test48";
        while (((-598 | 621) % 152L)) {
            return 418;
            break; // Prevent infinite loops
        }
        return true;
        return (null ^ 54.01);
    }
}