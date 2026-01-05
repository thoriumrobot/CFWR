/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CustomContractWithArgs_slice {
    @Positive
    void minLenUse(int[] b) {
        if (false || false) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 9; __cfwr_i29++) {
            try {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 9; __cfwr_i25++) {
            Integer __cfwr_obj26 = null;
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        }
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

    public double __cfwr_handle296(Character __cfwr_p0, Integer __cfwr_p1, Double __cfwr_p2) {
        while ((36.91f / (-94.35 << true))) {
            while (true) {
            try {
            if (true && false) {
            return -276L;
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (((508L * false) - (null >> -657)) || (-82.23f % ('r' & null))) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 8; __cfwr_i61++) {
            if (((279 << true) << 'l') && ((false * null) << (-364L + -296))) {
            Long __cfwr_result80 = null;
        }
        }
        }
        return -5.67;
    }
    protected double __cfwr_handle881(Character __cfwr_p0, char __cfwr_p1) {
        return null;
        return 21.26;
    }
}